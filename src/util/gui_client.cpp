#include "gui_client.h"

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <iostream>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace qbot {

using WsClient = websocketpp::client<websocketpp::config::asio_client>;

// Simple JSON builder (avoid heavy dependencies)
namespace json {

std::string escape(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c;
        }
    }
    return result;
}

// Simple JSON parser for responses
class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    std::string get_string(const std::string& key) {
        size_t key_pos = text_.find("\"" + key + "\"");
        if (key_pos == std::string::npos) return "";
        size_t colon = text_.find(':', key_pos);
        if (colon == std::string::npos) return "";
        size_t start = text_.find('"', colon + 1);
        if (start == std::string::npos) return "";
        size_t end = text_.find('"', start + 1);
        if (end == std::string::npos) return "";
        return text_.substr(start + 1, end - start - 1);
    }

    int get_int(const std::string& key) {
        size_t key_pos = text_.find("\"" + key + "\"");
        if (key_pos == std::string::npos) return 0;
        size_t colon = text_.find(':', key_pos);
        if (colon == std::string::npos) return 0;
        size_t start = colon + 1;
        while (start < text_.size() && (text_[start] == ' ' || text_[start] == '\t')) start++;
        return std::atoi(text_.c_str() + start);
    }

private:
    std::string text_;
};

} // namespace json

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

    std::ostringstream ss;
    ss << R"({"type":"start","player_names":[")"
       << json::escape(player1_name) << R"(",")"
       << json::escape(player2_name) << R"("]})";
    send_json(ss.str());
}

void GUIClient::send_gamestate(const StateNode& node, int current_player, float score) {
    if (!is_connected()) return;

    // Determine current player from node if not specified
    if (current_player < 0) {
        current_player = node.is_p1_to_move() ? 0 : 1;
    }

    // Determine winner
    std::string winner_str = "null";
    if (node.is_terminal()) {
        winner_str = (node.terminal_value > 0) ? "0" : "1";
    }

    // Build walls array
    std::ostringstream walls_ss;
    walls_ss << "[";
    bool first_wall = true;
    for (uint8_t r = 0; r < 8; r++) {
        for (uint8_t c = 0; c < 8; c++) {
            // Horizontal fences - need c < 7 for valid placement
            if (c < 7 && node.fences.has_h_fence(r, c)) {
                if (!first_wall) walls_ss << ",";
                walls_ss << R"({"x":)" << static_cast<int>(c)
                         << R"(,"y":)" << static_cast<int>(r)
                         << R"(,"orientation":"h"})";
                first_wall = false;
            }
            // Vertical fences - need r < 7 for valid placement
            if (r < 7 && node.fences.has_v_fence(r, c)) {
                if (!first_wall) walls_ss << ",";
                walls_ss << R"({"x":)" << static_cast<int>(c)
                         << R"(,"y":)" << static_cast<int>(r)
                         << R"(,"orientation":"v"})";
                first_wall = false;
            }
        }
    }
    walls_ss << "]";

    // Build full gamestate JSON
    std::ostringstream ss;
    ss << R"({"type":"gamestate","players":[)"
       << R"({"x":)" << static_cast<int>(node.p1.col)
       << R"(,"y":)" << static_cast<int>(node.p1.row)
       << R"(,"walls":)" << static_cast<int>(node.p1.fences)
       << R"(,"name":"Player1"},)"
       << R"({"x":)" << static_cast<int>(node.p2.col)
       << R"(,"y":)" << static_cast<int>(node.p2.row)
       << R"(,"walls":)" << static_cast<int>(node.p2.fences)
       << R"(,"name":"Player2"}],"walls":)"
       << walls_ss.str()
       << R"(,"current_player":)" << current_player
       << R"(,"score":)" << score
       << R"(,"winner":)" << winner_str
       << "}";

    send_json(ss.str());
}

std::optional<GUIClient::GUIMove> GUIClient::request_move(int player) {
    if (!is_connected()) return std::nullopt;

    // Send request
    std::ostringstream ss;
    ss << R"({"type":"request_move","player":)" << player << "}";
    if (!send_json(ss.str())) {
        return std::nullopt;
    }

    // Wait for response
    auto response = receive_json();
    if (!response) {
        last_error_ = "Timeout waiting for move";
        return std::nullopt;
    }

    json::Parser parser(*response);
    std::string type = parser.get_string("type");

    if (type == "quit") {
        GUIMove move{};
        move.type = GUIMove::Type::Quit;
        return move;
    }

    if (type == "move") {
        GUIMove move{};
        std::string move_type = parser.get_string("move_type");
        move.x = static_cast<uint8_t>(parser.get_int("x"));
        move.y = static_cast<uint8_t>(parser.get_int("y"));

        if (move_type == "pawn") {
            move.type = GUIMove::Type::Pawn;
        } else if (move_type == "wall") {
            move.type = GUIMove::Type::Wall;
            std::string orient = parser.get_string("orientation");
            move.horizontal = (orient == "h");
        }
        return move;
    }

    last_error_ = "Unexpected message type: " + type;
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
