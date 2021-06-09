#include "Label.cpp"

class Poliz
{
    Lex *p;
    int size;
    int free;
public:
    Poliz ( int maxSize = 1000 )
    {
        p = new Lex[size = maxSize];
        free = 0;
    };

    ~Poliz() {
        delete []p;
    };

    void putLex(Lex l) {
        p[free]=l; ++free;
    }

    void putLex(Lex l, int place) {
        p[place]=l;
    }

    void blank() {
        free++;
    }

    int getFree() {
        return free;
    }

    Lex& operator[] (int index ) {
        if (index > size) {
            throw "POLIZ:out of array";
        } else if ( index > free ) {
            throw "POLIZ:indefinite element of array";
        } else {
            return p[index];
        }
    }

    void print() {
        for ( int i = 0; i < free; i++ )
            cout << p[i] << endl;
    }
};
