#pragma once

#include "../tree/StateNode.h"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

namespace qbot {

/// WebSocket client for communicating with the standalone Quoridor GUI.
///
/// Provides visualization of game states and input handling from the GUI.
/// Connection is optional - if not connected, visualization falls back to print_node().
///
/// Thread safety: connect/disconnect should be called from one thread.
/// send_gamestate and receive_move can be called from any thread once connected.
class GUIClient {
public:
    /// Move received from GUI
    struct GUIMove {
        enum class Type { Pawn, Wall, Quit };
        Type type;
        uint8_t x;           // column
        uint8_t y;           // row
        bool horizontal;     // for wall moves only
    };

    /// Connection configuration
    struct Config {
        std::string host{"localhost"};
        uint16_t port{8765};
        int connect_timeout_ms{5000};
        int read_timeout_ms{30000};  // timeout for waiting on moves
    };

    GUIClient();
    ~GUIClient();

    // Non-copyable, non-movable
    GUIClient(const GUIClient&) = delete;
    GUIClient& operator=(const GUIClient&) = delete;
    GUIClient(GUIClient&&) = delete;
    GUIClient& operator=(GUIClient&&) = delete;

    /// Connect to GUI server
    /// @param config Connection configuration
    /// @return true if connected successfully
    bool connect(const Config& config);

    /// Disconnect from GUI server
    void disconnect();

    /// Check if connected
    [[nodiscard]] bool is_connected() const noexcept;

    /// Send a game state to the GUI for visualization
    /// @param node The state to visualize
    /// @param current_player Whose turn it is (0 or 1)
    /// @param score Optional evaluation score to display
    void send_gamestate(const StateNode& node, int current_player, float score = 0.0f);

    /// Send start message to initialize game with player names
    void send_start(const std::string& player1_name, const std::string& player2_name);

    /// Request a move from the GUI (blocking)
    /// @param player Which player should move (0 = human at GUI)
    /// @return The move made, or std::nullopt on error/quit
    [[nodiscard]] std::optional<GUIMove> request_move(int player);

    /// Convert a GUIMove to the engine's Move type
    /// @param gui_move Move from GUI
    /// @return Engine move, or invalid Move if quit
    [[nodiscard]] static Move to_engine_move(const GUIMove& gui_move);

    /// Get the last error message
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_error_;
    std::atomic<bool> connected_{false};

    bool send_json(const std::string& json);
    std::optional<std::string> receive_json();
};

/// Visualize a StateNode - uses GUI if available, otherwise prints to stdout
/// @param node The state to display
/// @param gui Optional GUI client (can be nullptr)
/// @param current_player Whose turn it is
/// @param score Evaluation score
void visualize_state(const StateNode& node, GUIClient* gui = nullptr,
                     int current_player = -1, float score = 0.0f);

} // namespace qbot
