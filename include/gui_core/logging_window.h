#ifndef COAST_LOGGING_WINDOW_H
#define COAST_LOGGING_WINDOW_H

namespace coast {

/**
 * Window for viewing application logs.
 * Provides a scrollable view of log messages with filtering options.
 *
 * Currently a placeholder - future implementation will include:
 * - Log level filtering
 * - Search functionality
 * - Auto-scroll toggle
 * - Copy to clipboard
 */
class logging_window {
public:
    /**
     * Render the logging window.
     */
    void render();

    /**
     * Access visibility state for menu toggle.
     */
    bool& visible() { return is_visible; }

private:
    bool is_visible = false;
};

} // namespace coast

#endif // COAST_LOGGING_WINDOW_H
