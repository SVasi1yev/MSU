#include <cstring>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
const int bufSize = 130;

enum LexType {
    LEX_NULL,		//0
    LEX_PROGRAM,	//1
    LEX_END,		//2

    LEX_STRUCT,		//3

    LEX_T_INT,		//4
    LEX_T_STRING,	//5
    LEX_T_BOOLEAN,	//6
    LEX_NUM,		//7
    LEX_STRING,		//8
    LEX_TRUE,		//9
    LEX_FALSE,		//10

    LEX_IF,			//11
    LEX_ELSE,		//12

    LEX_FOR,		//13
    LEX_WHILE,		//14
    LEX_BREAK,		//15

    LEX_AND,		//16
    LEX_OR,			//17
    LEX_NOT,		//18

    LEX_READ,		//19
    LEX_WRITE,		//20

    LEX_SEMICOLON,	//21
    LEX_COMMA,		//22
    LEX_COLON,		//23
    LEX_ASSIGN,		//24

    LEX_LPAREN,		//25
    LEX_RPAREN,		//26
    LEX_LBRACE,		//27
    LEX_RBRACE,		//28

    LEX_PLUS,		//29
    LEX_MINUS,		//30
    LEX_TIMES,		//31
    LEX_SLASH,		//32

    LEX_EQ,			//33
    LEX_LSS,		//34
    LEX_GTR,		//35
    LEX_LEQ,		//36
    LEX_NEQ,		//37
    LEX_GEQ,		//38

    LEX_ID,			//39
    LEX_QUOTE,      //40
    LEX_GOTO,       //41
    LEX_DOT,        //42

    POLIZ_LABEL,    //43
    POLIZ_ADDRESS,  //44
    POLIZ_GO,       //45
    POLIZ_FGO,      //46

    LEX_UN_MINUS,   //47
    LEX_T_STRUCT,    //48
    POLIZ_STRUCT_ID,   //49
    POLIZ_STRUCT_FIELD_ID, //50
    POLIZ_VAR_ID//51
};

enum States {
    ST_S,
    ST_IDorSW,
    ST_NUM,
    ST_COM0,
    ST_COM1,
    ST_COM2,
    ST_STR,
    ST_COMP,
    ST_NEQ,
    ST_OTHER,
    ST_ERR,
    ST_END
};

class Lex {
    LexType lexType;
    int intLexValue;
    char* strLexValue;
    string strLexValue1;
    static const char* lexTypeString[];
public:
    Lex (LexType t = LEX_NULL, int v = 0, const char* str = "", string str1 = "") {
        lexType = t;
        intLexValue = v;
        strLexValue = new char[strlen(str) + 1];
        strcpy(strLexValue, str);
        strLexValue1 = str1;
    }

    ~Lex() {
       // delete[] strLexValue;
    }

    LexType getLexType () const {return lexType;}

    int getIntLexValue () const {return intLexValue;}

    const char* getStrLexValue () const {return strLexValue;}

    string getStrLexValue1 () const {return strLexValue1;}

    static const char* lexTypeToString (LexType lexType) {
        return lexTypeString[lexType];
    }

    friend ostream& operator << (ostream &s, Lex l) {
        s << "(" << lexTypeToString(l.lexType)
          << ", " << l.intLexValue
          << ", " << '"' << l.strLexValue
          << '"' << ", " << '"' << l.strLexValue1
          << '"' << ");";
        return s;
    }
};

const char* Lex::lexTypeString[] = {
    "LEX_NULL",		//0
    "LEX_PROGRAM",	//1
    "LEX_END",		//2

    "LEX_STRUCT",	//3

    "LEX_T_INT",	//4
    "LEX_T_STRING",	//5
    "LEX_T_BOOLEAN",//6
    "LEX_NUM",		//7
    "LEX_STRING",	//8
    "LEX_TRUE",		//9
    "LEX_FALSE",	//10

    "LEX_IF",		//11
    "LEX_ELSE",		//12

    "LEX_FOR",		//13
    "LEX_WHILE",	//14
    "LEX_BREAK",	//15

    "LEX_AND",		//16
    "LEX_OR",		//17
    "LEX_NOT",		//18

    "LEX_READ",		//19
    "LEX_WRITE",	//20

    "LEX_SEMICOLON",//21
    "LEX_COMMA",	//22
    "LEX_COLON",	//23
    "LEX_ASSIGN",	//24

    "LEX_LPAREN",	//25
    "LEX_RPAREN",   //26
    "LEX_LBRACE",	//27
    "LEX_RBRACE",	//28

    "LEX_PLUS",		//29
    "LEX_MINUS",	//30
    "LEX_TIMES",	//31
    "LEX_SLASH",	//32

    "LEX_EQ",		//33
    "LEX_LSS",		//34
    "LEX_GTR",		//35
    "LEX_LEQ",		//36
    "LEX_NEQ",		//37
    "LEX_GEQ",		//38

    "LEX_ID",   	//39
    "LEX_QUOTE",    //40
    "LEX_GOTO",     //41
    "LEX_DOT",       //42

    "POLIZ_LABEL",  //43
    "POLIZ_ADDRESS",//44
    "POLIZ_GO",     //45
    "POLIZ_FGO",    //46

    "LEX_UN_MINUS", //47
    "LEX_T_STRUCT",    //48
    "POLIZ_STRUCT_ID",   //49
    "POLIZ_STRUCT_FIELD_ID", //50
    "POLIZ_VAR_ID"  //51
};

class LexScanner {
    States state;
    ifstream file;
    char* program;
    char c;
    char buf[bufSize];
    int bufTop;
    bool _eof;
    int rowCount;
    int charCount;

    void bufClear () {
        bufTop = 0;
        for (int i = 0; i < bufSize; i++) {
            buf[i] = '\0';
        }
    }

    void bufAdd () {
        buf[bufTop++] = c;
        if (bufTop >= bufSize - 1) {
            state = ST_ERR;
        }
    }

    bool fileOpen () {
        return file.is_open();
    }

    void getChar () {
        c = file.get();
        if (c == -1) {
            _eof = true;
            rowCount++;
            charCount = 0;
        } else if (c == '\n') {
            rowCount++;
            charCount = 0;
        } else {
            charCount++;
        }
    }

    int look(const char* str) {
        for(int i = 0; SL[i] != NULL; i++) {
            if(strcmp(SL[i], str) == 0) {
                return i;
            }
        }
        return 0;
    }

    bool isLetter () {
        return (c >= 'a') && (c <= 'z')
                || (c >= 'A') && (c <= 'Z');
    }

    bool isDigit () {
        return (c >= '0') && (c <= '9');
    }

    bool isInsign () {
        return (c == ' ') || (c == '\n')
                || (c == '\t') || (c == '\r');
    }

    bool isOther () {
        return (c == '+') || (c == '-')
                || (c == '*') || (c == ',')
                || (c == ';') || (c == ':')
                || (c == '.') || (c == '(')
                || (c == ')') || (c == '{')
                || (c == '}');
    }

    bool isComp () {
        return (c == '=') || (c == '<') || (c == '>');
    }

    bool eof () {
        return _eof;
    }

public:
    static const char* SL[];
    static const LexType LSL[];

    LexScanner (const char* program): rowCount(1), charCount(1) {
        this->program = new char[100];
        strcpy(this->program, program);
        file.open(program, ios::in);
        resetFile();
        state = ST_S;
        c = ' ';
        bufClear();
        _eof = false;
    }

    ~LexScanner () {
        file.close();
    }

    void resetFile () {
        file.close();
        file.open(program, ios::in);
        rowCount = 1;
        charCount = 1;
        state = ST_S;
        c = ' ';
        bufClear();
        _eof = false;
    }

    void showRowChar () {
        cout << "row: " << rowCount
             << "; char: " << charCount << ";\n";
    }

    Lex getLex();
};

const char* LexScanner::SL[] = {
    "",
    "program",
    "struct",
    "int",
    "string",
    "boolean",
    "true",
    "false",
    "if",
    "else",
    "for",
    "while",
    "break",
    "and",
    "or",
    "not",
    "read",
    "write",
    "goto",

    ";",
    ",",
    ":",
    ".",
    "=",
    "(",
    ")",
    "{",
    "}",
    "+",
    "-",
    "*",
    "/",
    "==",
    "<",
    ">",
    "<=",
    "!=",
    ">=",
    "\"",

    NULL
};

const LexType LexScanner::LSL[] = {
    LEX_NULL,

    LEX_PROGRAM,
    LEX_STRUCT,
    LEX_T_INT,
    LEX_T_STRING,
    LEX_T_BOOLEAN,
    LEX_TRUE,
    LEX_FALSE,
    LEX_IF,
    LEX_ELSE,
    LEX_FOR,
    LEX_WHILE,
    LEX_BREAK,
    LEX_AND,
    LEX_OR,
    LEX_NOT,
    LEX_READ,
    LEX_WRITE,
    LEX_GOTO,

    LEX_SEMICOLON,
    LEX_COMMA,
    LEX_COLON,
    LEX_DOT,
    LEX_ASSIGN,
    LEX_LPAREN,
    LEX_RPAREN,
    LEX_LBRACE,
    LEX_RBRACE,
    LEX_PLUS,
    LEX_MINUS,
    LEX_TIMES,
    LEX_SLASH,
    LEX_EQ,
    LEX_LSS,
    LEX_GTR,
    LEX_LEQ,
    LEX_NEQ,
    LEX_GEQ,

    LEX_NULL
};

Lex LexScanner::getLex() {
    int n = 0, k = 0;
    state = ST_S;
    while(true) {
        switch (state) {
            case ST_S:
                if (isInsign()) {
                    getChar();
                } else if (isLetter()) {
                    bufClear();
                    bufAdd();
                    getChar();
                    state = ST_IDorSW;
                } else if (isDigit()) {
                    n = c - '0';
                    getChar();
                    state = ST_NUM;
                } else if (c == '/') {
                    getChar();
                    state = ST_COM0;
                } else if (c == '"') {
                    bufClear();
                    getChar();
                    state = ST_STR;
                } else if (isComp()) {
                    bufClear();
                    bufAdd();
                    getChar();
                    state = ST_COMP;
                } else if (c == '!') {
                    getChar();
                    state = ST_NEQ;
                } else if (isOther()) {
                    bufClear();
                    bufAdd();
                    k = look(buf);
                    getChar();
                    return Lex(LSL[k], k, SL[k]);
                } else if (eof()) {
                    return Lex(LEX_END, 0, "");
                } else {
                    getChar();
                    state = ST_ERR;
                }
                break;

            case ST_IDorSW:
                if (isDigit() || isLetter()) {
                    bufAdd();
                    getChar();
                } else {
                    k = look(buf);
                    if (k == 0) {
                        //k = idTable.put(buf);
                        return Lex(LEX_ID, k, buf);
                    } else {
                        return Lex(LSL[k], k, SL[k]);
                    }
                }
                break;

            case ST_NUM:
                if (isDigit()) {
                    n = n * 10 + (c - '0');
                    getChar();
                } else {
                    return Lex(LEX_NUM, n, "");
                }
                break;

            case ST_COM0:
                if (c == '*') {
                    getChar();
                    state = ST_COM1;
                } else {
                    k = look("/");
                    return Lex(LSL[k], k, SL[k]);
                }
                break;

            case ST_COM1:
                if (c == '*') {
                    getChar();
                    state = ST_COM2;
                } else if (eof()) {
                    state = ST_ERR;
                } else {
                    getChar();
                }
                break;

            case ST_COM2:
                if (c == '/') {
                    getChar();
                    state = ST_S;
                } else if (eof()) {
                    state = ST_ERR;
                } else {
                    getChar();
                    state = ST_COM1;
                }
                break;

            case ST_STR:
                if (c == '"') {
                    getChar();
                    return Lex(LEX_STRING, 0, buf);
                } else if (eof()) {
                    state = ST_ERR;
                } else {
                    bufAdd();
                    getChar();
                }
                break;

            case ST_COMP:
                if (c == '=') {
                    bufAdd();
                    getChar();
                    k = look(buf);
                    return Lex(LSL[k], k, SL[k]);
                } else {
                    k = look(buf);
                    return Lex(LSL[k], k, SL[k]);
                }
                break;

            case ST_NEQ:
                if (c == '=') {
                    k = look("!=");
                    getChar();
                    return Lex(LSL[k], k, SL[k]);
                } else {
                    state = ST_ERR;
                }
                break;

            case ST_ERR:
                cout << "lex_error!" << " - line: " << rowCount
                     << " - char: " << charCount - 1 << endl;
                resetFile();
                throw "lex error!";
                break;
        }
    }
}