
namespace readKeyboard
{
    #ifdef _WIN32
        #include <conio.h>
    #else
        #include <unistd.h>
        #include <termios.h>
        #include <fcntl.h>

        void setNonBlocking(bool enable) {
            struct termios ttystate;
            tcgetattr(STDIN_FILENO, &ttystate);
            if (enable) {
                ttystate.c_lflag &= ~(ICANON | ECHO);
                tcsetattr(STDIN_FILENO, TCSANOW, &ttystate);
                fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
            } else {
                ttystate.c_lflag |= ICANON | ECHO;
                tcsetattr(STDIN_FILENO, TCSANOW, &ttystate);
                fcntl(STDIN_FILENO, F_SETFL, 0);
            }
        }

        bool kbhit() {
            char ch;
            int nread;
            nread = read(STDIN_FILENO, &ch, 1);
            if (nread == 1) {
                ungetc(ch, stdin);
                return true;
            }
            return false;
        }

        char getch() {
            return getchar();
        }
    #endif

    char pollKey() {
    #ifdef _WIN32
        if (_kbhit())
            return _getch();
    #else
        if (kbhit())
            return getch();
    #endif
        return '\0';  // nic nie naciśnięto
    }

void setupKeyboard() {
    #ifdef __linux__
        setNonBlocking(true);
    #endif
}

void cleanupKeyboard() {
    #ifdef __linux__
        setNonBlocking(false);
    #endif
}
}
