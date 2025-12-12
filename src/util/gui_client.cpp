#include "gui_client.h"

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>

#include <iostream>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace qbot {

using WsClient = websocketpp::client<websocketpp::config::asio_client>;
using json = nlohmann::json;

struct GUIClient::Impl {
    WsClient client;
    websocketpp::connection_hdl connection;
    std::thread io_thread;

    std::mutex msg_mutex;
    std::condition_variable msg_cv;
    std::queue<std::string> incoming_messages;

    bool connected{false};
    std::string error_msg;

    Config config;

    Impl() {
        client.clear_access_channels(websocketpp::log::alevel::all);
        client.clear_error_channels(websocketpp::log::elevel::all);

        client.init_asio();

        client.set_open_handler([this](websocketpp::connection_hdl) {
            std::lock_guard<std::mutex> lock(msg_mutex);
            connected = true;
            msg_cv.notify_all();
        });

        client.set_fail_handler([this](websocketpp::connection_hdl) {
            std::lock_guard<std::mutex> lock(msg_mutex);
            connected = false;
            error_msg = "Connection failed";
            msg_cv.notify_all();
        });

        client.set_close_handler([this](websocketpp::connection_hdl) {
            std::lock_guard<std::mutex> lock(msg_mutex);
            connected = false;
            msg_cv.notify_all();
        });

        client.set_message_handler([this](websocketpp::connection_hdl, WsClient::message_ptr msg) {
            std::lock_guard<std::mutex> lock(msg_mutex);
            incoming_messages.push(msg->get_payload());
            msg_cv.notify_all();
        });
    }

    ~Impl() {
        stop();
    }

    void stop() {
        if (connected) {
            try {
                client.close(connection, websocketpp::close::status::going_away, "");
            } catch (...) {}
        }

        client.stop();

        if (io_thread.joinable()) {
            io_thread.join();
        }
        connected = false;
    }
};

GUIClient::GUIClient() : impl_(std::make_unique<Impl>()) {}

GUIClient::~GUIClient() {
    disconnect();
}

bool GUIClient::connect(const Config& config) {
    if (connected_) {
        disconnect();
    }

    impl_->config = config;
    impl_->error_msg.clear();

    std::string uri = "ws://" + config.host + ":" + std::to_string(config.port);

    websocketpp::lib::error_code ec;
    WsClient::connection_ptr con = impl_->client.get_connection(uri, ec);

    if (ec) {
        last_error_ = "Connection error: " + ec.message();
        return false;
    }

    impl_->connection = con->get_handle();
    impl_->client.connect(con);

    // Start IO thread
    impl_->io_thread = std::thread([this]() {
        impl_->client.run();
    });

    // Wait for connection with timeout
    {
        std::unique_lock<std::mutex> lock(impl_->msg_mutex);
        bool success = impl_->msg_cv.wait_for(
            lock,
            std::chrono::milliseconds(config.connect_timeout_ms),
            [this]() { return impl_->connected || !impl_->error_msg.empty(); }
        );

        if (!success || !impl_->connected) {
            last_error_ = impl_->error_msg.empty() ? "Connection timeout" : impl_->error_msg;
            impl_->stop();
            return false;
        }
    }

    connected_ = true;
    return true;
}

void GUIClient::disconnect() {
    impl_->stop();
    connected_ = false;
}

bool GUIClient::is_connected() const noexcept {
    return connected_ && impl_->connected;
}

bool GUIClient::send_json(const std::string& json_str) {
    if (!is_connected()) return false;

    websocketpp::lib::error_code ec;
    impl_->client.send(impl_->connection, json_str, websocketpp::frame::opcode::text, ec);

    if (ec) {
        last_error_ = "Send error: " + ec.message();
        return false;
    }
    return true;
}

std::optional<std::string> GUIClient::receive_json() {
    if (!is_connected()) return std::nullopt;

    std::unique_lock<std::mutex> lock(impl_->msg_mutex);
    bool success = impl_->msg_cv.wait_for(
        lock,
        std::chrono::milliseconds(impl_->config.read_timeout_ms),
        [this]() { return !impl_->incoming_messages.empty() || !impl_->connected; }
    );

    if (!success || impl_->incoming_messages.empty()) {
        return std::nullopt;
    }

    std::string msg = std::move(impl_->incoming_messages.front());
    impl_->incoming_messages.pop();
    return msg;
}

void GUIClient::send_start(const std::string& player1_name, const std::string& player2_name) {
    if (!is_connected()) return;

    json msg = {
        {"type", "start"},
        {"player_names", {player1_name, player2_name}}
    };
    send_json(msg.dump());
}

void GUIClient::send_gamestate(const StateNode& node, int current_player, float score) {
    if (!is_connected()) return;

    // Determine current player from node if not specified
    if (current_player < 0) {
        current_player = node.is_p1_to_move() ? 0 : 1;
    }

    // Build walls array
    json walls = json::array();
    for (uint8_t r = 0; r < 8; r++) {
        for (uint8_t c = 0; c < 8; c++) {
            if (c < 7 && node.fences.has_h_fence(r, c)) {
                walls.push_back({{"x", c}, {"y", r}, {"orientation", "h"}});
            }
            if (r < 7 && node.fences.has_v_fence(r, c)) {
                walls.push_back({{"x", c}, {"y", r}, {"orientation", "v"}});
            }
        }
    }

    // Build full gamestate JSON
    json msg = {
        {"type", "gamestate"},
        {"players", {
            {{"x", node.p1.col}, {"y", node.p1.row}, {"walls", node.p1.fences}, {"name", "Player1"}},
            {{"x", node.p2.col}, {"y", node.p2.row}, {"walls", node.p2.fences}, {"name", "Player2"}}
        }},
        {"walls", walls},
        {"current_player", current_player},
        {"score", score},
        {"winner", node.is_terminal() ? json(node.terminal_value > 0 ? 0 : 1) : json(nullptr)}
    };

    send_json(msg.dump());
}

std::optional<GUIClient::GUIMove> GUIClient::request_move(int player) {
    if (!is_connected()) return std::nullopt;

    // Send request
    json request = {{"type", "request_move"}, {"player", player}};
    if (!send_json(request.dump())) {
        return std::nullopt;
    }

    // Wait for response
    auto response = receive_json();
    if (!response) {
        last_error_ = "Timeout waiting for move";
        return std::nullopt;
    }

    try {
        json msg = json::parse(*response);
        std::string type = msg.value("type", "");

        if (type == "quit") {
            GUIMove move{};
            move.type = GUIMove::Type::Quit;
            return move;
        }

        if (type == "move") {
            GUIMove move{};
            std::string move_type = msg.value("move_type", "");
            move.x = msg.value("x", 0);
            move.y = msg.value("y", 0);

            if (move_type == "pawn") {
                move.type = GUIMove::Type::Pawn;
            } else if (move_type == "wall") {
                move.type = GUIMove::Type::Wall;
                std::string orient = msg.value("orientation", "");
                move.horizontal = (orient == "h");
            }
            return move;
        }

        last_error_ = "Unexpected message type: " + type;
    } catch (const json::exception& e) {
        last_error_ = std::string("JSON parse error: ") + e.what();
    }

    return std::nullopt;
}

Move GUIClient::to_engine_move(const GUIMove& gui_move) {
    switch (gui_move.type) {
        case GUIMove::Type::Pawn:
            return Move::pawn(gui_move.y, gui_move.x);
        case GUIMove::Type::Wall:
            return Move::fence(gui_move.y, gui_move.x, gui_move.horizontal);
        case GUIMove::Type::Quit:
        default:
            return Move{};  // Invalid move
    }
}

void visualize_state(const StateNode& node, GUIClient* gui, int current_player, float score) {
    if (gui && gui->is_connected()) {
        gui->send_gamestate(node, current_player, score);
    } else {
        node.print_node();
    }
}

} // namespace qbot
