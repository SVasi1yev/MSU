#include <cstring>
#include <iostream>
#include <map>
#include <stack>
#include "Poliz.cpp"

using namespace std;

struct lab {
    string name;
    int place;
};

class Parser {
    Lex curLex;
    LexType curType;
    int curIntLexValue;
    string curStrLexValue;
    LexScanner scanner;

    int deep;
    bool onlyId;
    bool idInRead;
    LexType idType;
    int breakCount;
    int prevBreakCount;

    stack<LexType> opStack;
    stack<string> structStack;
    stack<lab> labelStack;
    stack<int> breakStack;

    map <string, Variable> tempMap;
    LexType tempType;
    string tempStr;
    string tempStr1;
    string tempStr2;
    Variable tempVar;
    Label tempLabel;
    Lex tempLex;
    int tempInt;
    lab tempLab;

    void getLex() {
        curLex = scanner.getLex();
        curType = curLex.getLexType();
        curIntLexValue = curLex.getIntLexValue();
        curStrLexValue = curLex.getStrLexValue();
        //cout << curLex << endl;
    }

    void P();
    void S();
    void S1();
    void S2();
    void S3();
    void D();
    void D1();
    void D2();
    void D3();
    void D4();
    void O();
    void O1();
    void I();
    void E();
    void E1();
    void E2();
    void E3();
    void E4();
    void E5();
    void E6();
    void Z();



    void checkOp();
    void checkNot();
    void checkUnMinus();
    void eqType();
    void exIsBool();
public:
    map <string, map<string, Variable>> structsMap;
    map <string, Variable> variablesMap;
    map <string, Label> labelMap;

    Poliz prog;

    Parser (const char* program): scanner(program), prog(1000) {}

    LexScanner& getScanner () {
        return scanner;
    }

    void analyze();
    void execute ();

    Poliz& getPoliz() {
        return prog;
    }

    void printVariables () {
        map <string, Variable>::iterator it;

        for (it = variablesMap.begin(); it != variablesMap.end(); it++) {
            cout << it->first << "\t:\t" << it->second << endl;
            if ((it->second).getLexType() == LEX_T_STRUCT) {
                map <string, Variable>::iterator it1;
                for (it1 = (it->second).getStructFields().begin(); it1 != (it->second).getStructFields().end(); it1++) {
                    cout << it1->first << "\t:\t" << it1->second << endl;
                }
            }
            cout << endl;
        }
    }

    void printLabels () {
        map <string, Label>::iterator it;

        for (it = labelMap.begin(); it != labelMap.end(); it++) {
            cout << it->first << "\t:\t" << it->second << endl;
        }
        cout << endl;
    }

    void printStructs () {
        map <string, map<string, Variable>>::iterator it1;
        map <string, Variable>::iterator it2;

        for (it1 = structsMap.begin(); it1 != structsMap.end(); it1++) {
            cout << it1->first << ":" << endl;
            for (it2 = (it1->second).begin(); it2 != (it1->second).end(); it2++) {
                cout << it2->first << "\t:\t" << it2->second << endl;
            }
            cout << endl;
        }
    }

    void printPoliz () {
        prog.print();
    }
};

void Parser::checkOp() {
    cout << "checkOp\n";
    LexType t1, t2, op, r;
    t2 = opStack.top();
    opStack.pop();
    op = opStack.top();
    opStack.pop();
    t1 = opStack.top();
    opStack.pop();
    if ((op == LEX_EQ) || (op == LEX_LSS) || (op == LEX_GTR) || (op == LEX_NEQ)) {
        r = LEX_T_BOOLEAN;
        if ((t1 != LEX_T_INT) && (t1 != LEX_T_STRING) || (t1 != t2)) {
            throw curLex;
        }
    } else if ((op == LEX_GEQ) || (op == LEX_LSS)) {
        r = LEX_T_BOOLEAN;
        if ((t1 != LEX_T_INT) || (t1 != t2)) {
            throw curLex;
        }
    } else if ((op == LEX_OR) || (op == LEX_AND)) {
        r = LEX_T_BOOLEAN;
        if ((t1 != LEX_T_BOOLEAN) || (t1 != t2)) {
            throw curLex;
        }
    } else if (op == LEX_PLUS) {
        if ((t1 == LEX_T_INT) && (t1 == t2)) {
            r = LEX_T_INT;
        } else if ((t1 == LEX_T_STRING) && (t1 == t2)) {
            r = LEX_T_STRING;
        } else {
            throw curLex;
        }
    } else if ((op == LEX_MINUS) || (op == LEX_TIMES) || (op == LEX_SLASH)) {
        r = LEX_T_INT;
        if ((t1 != LEX_T_INT) || (t1 != t2)) {
            throw curLex;
        }
    } else {
        throw curLex;
    }
    opStack.push(r);
    prog.putLex(Lex(op));
}

void Parser::checkNot() {
    cout << "checkNot\n";
    if (opStack.top() != LEX_T_BOOLEAN) {
        throw curLex;
    }
    prog.putLex(Lex(LEX_NOT));
}

void Parser::checkUnMinus() {
    cout << "checkNot\n";
    if (opStack.top() != LEX_T_INT) {
        throw curLex;
    }
    prog.putLex(Lex(LEX_UN_MINUS));
}

void Parser::exIsBool() {
    cout << "exIsBool\n";
    if (opStack.top() != LEX_T_BOOLEAN) {
        throw curLex;
    }
    opStack.pop();
}

void Parser::analyze() {
    scanner.resetFile();
    getLex();
    while (curType != LEX_END) {
        tempLex = curLex;
        getLex();
        if ((tempLex.getLexType() == LEX_ID) && (curType == LEX_COLON)) {
            if (labelMap.count(tempLex.getStrLexValue()) != 0) {
                throw curLex;
            }
            tempLabel = Label(tempLex.getStrLexValue());
            labelMap.insert(pair<string, Label>(tempLex.getStrLexValue(), tempLabel));
        }
    }
    scanner.resetFile();
    deep = 0;
    prevBreakCount = 0;
    breakCount = 0;
    onlyId = false;
    idInRead = false;
    getLex();
    P();
    cout << "GOOD!" << endl;
}

void Parser::P() {
    cout << "P\n";
    if (curType != LEX_PROGRAM) {
        throw curLex;
    }
    getLex();
    if (curType != LEX_LBRACE) {
        throw curLex;
    }
    getLex();
    S();
    D();
    O();

    if (curType != LEX_RBRACE) {
        throw curLex;
    }

    while (!labelStack.empty()) {
        lab tempLab = labelStack.top();
        labelStack.pop();
        prog.putLex(Lex(POLIZ_LABEL, (labelMap.find(tempLab.name)->second).getPlace()), tempLab.place);
    }

    getLex();
    if (curType != LEX_END) {
        throw curLex;
    }
}

void  Parser::S() {
    cout <<"S\n";
    while (curType == LEX_STRUCT) {
        getLex();
        if ((curType != LEX_ID)
            || (structsMap.count(curStrLexValue) != 0)) {
            throw curLex;
        }
        tempStr = curStrLexValue;
        getLex();
        if (curType != LEX_LBRACE) {
            throw curLex;
        }
        getLex();
        S1();
        if (curType != LEX_RBRACE) {
            throw curLex;
        }
        getLex();
        if (curType != LEX_SEMICOLON) {
            throw curLex;
        }
        structsMap.insert(pair<string, map<string, Variable>>(tempStr, tempMap));
        tempMap.clear();
        getLex();
    }
}

void Parser::S1() {
    cout <<"S1\n";
    if ((curType != LEX_T_INT) && (curType != LEX_T_STRING)
        && (curType != LEX_T_BOOLEAN)) {
        throw curLex;
    }
    tempType = curType;
    getLex();
    if ((curType != LEX_ID)
        || (tempMap.count(curStrLexValue) != 0)) {
        throw curLex;
    }
    tempMap.insert(pair<string, Variable>(curStrLexValue, Variable(curStrLexValue, tempType)));
    getLex();
    S2();
    if (curType != LEX_SEMICOLON) {
        throw curLex;
    }
    getLex();
    S3();
}

void Parser::S2() {
    cout << "S2\n";
    while (curType == LEX_COMMA) {
        getLex();
        if ((curType != LEX_ID)
            || (tempMap.count(curStrLexValue) != 0)) {
            throw curLex;
        }
        tempMap.insert(pair<string, Variable>(curStrLexValue, Variable(curStrLexValue, tempType)));
        getLex();
    }
}

void Parser::S3() {
    cout << "S3\n";
    while ((curType == LEX_T_INT) || (curType == LEX_T_STRING)
        || (curType == LEX_T_BOOLEAN)) {
        tempType = curType;
        getLex();
        if ((curType != LEX_ID)
            || (tempMap.count(curStrLexValue) != 0)) {
            throw curLex;
        }
        tempMap.insert(pair<string, Variable>(curStrLexValue, Variable(curStrLexValue, tempType)));
        getLex();
        S2();
        if (curType != LEX_SEMICOLON) {
            throw curLex;
        }
        getLex();
    }
}

void Parser::D() {
    cout <<"D\n";
    while ((curType == LEX_T_INT) || (curType == LEX_T_STRING) || (curType == LEX_T_BOOLEAN)
           || ((curType == LEX_ID) && (structsMap.count(curLex.getStrLexValue()) != 0))) {
        if ((curType == LEX_ID) && (structsMap.count(curLex.getStrLexValue()) != 0)) {
            tempType = LEX_T_STRUCT;
            tempStr = curStrLexValue;
            tempMap = (structsMap.find(tempStr)->second);
            getLex();
            D4();
        } else {
            tempType = curType;
            getLex();
            D1();
            D2();
        }
        if (curType != LEX_SEMICOLON) {
            throw curLex;
        }
        getLex();
    }
}

void Parser::D1() {
    cout <<"D1\n";
    if ((curType != LEX_ID)
        || (variablesMap.count(curStrLexValue) != 0)) {
        throw curLex;
    }
    tempStr = curStrLexValue;
    tempVar = Variable(curStrLexValue, tempType);
    getLex();
    D3();
    variablesMap.insert(pair<string, Variable>(tempStr, tempVar));
}

void Parser::D2() {
    cout <<"D2\n";
    while (curType == LEX_COMMA) {
        getLex();
        D1();
    }
}

void Parser::D3() {
    cout <<"D3\n";
    if (curType == LEX_ASSIGN) {
        tempVar.setAssign(true);
        getLex();
        if (curType == LEX_MINUS) {
            if (tempType != LEX_T_INT) {
                throw curLex;
            }
            curType == LEX_MINUS ? tempInt = -1 : tempInt = 1;
            getLex();
            if (curType != LEX_NUM) {
                throw curLex;
            }
            tempInt *= curIntLexValue;
            tempVar.setIntValue(tempInt);
            getLex();
        } else if (curType == LEX_NUM) {
            if (tempType != LEX_T_INT) {
                throw curLex;
            }
            tempVar.setIntValue(curIntLexValue);
            getLex();
        } else if (curType == LEX_STRING) {
            if (tempType != LEX_T_STRING) {
                throw curLex;
            }
            tempVar.setStrValue(curStrLexValue);
            getLex();
        } else if ((curType == LEX_TRUE) || (curType == LEX_FALSE)) {
            if (tempType != LEX_T_BOOLEAN) {
                throw curLex;
            }
            curType == LEX_TRUE ? tempVar.setIntValue(1) : tempVar.setIntValue(0);
            getLex();
        } else {
            throw curLex;
        }
    }
}

void Parser::D4() {
    cout << "D4\n";
    if ((curType != LEX_ID)
        || (variablesMap.count(curStrLexValue) != 0)) {
        throw curLex;
    }
    tempVar = Variable(curStrLexValue, LEX_T_STRUCT, 0, "", tempStr, tempMap);
    variablesMap.insert(pair<string, Variable>(curStrLexValue, tempVar));
    getLex();
    while (curType == LEX_COMMA) {
        getLex();
        if ((curType != LEX_ID)
            || (variablesMap.count(curStrLexValue) != 0)) {
            throw curLex;
        }
        tempVar = Variable(curStrLexValue, LEX_T_STRUCT, 0, "", tempStr, tempMap);
        variablesMap.insert(pair<string, Variable>(curStrLexValue, tempVar));
        getLex();
    }
}

void Parser::O() {
    cout << "O\n";
    while (curType != LEX_RBRACE) {
        O1();
    }
}

void Parser::O1() {
    cout << "O1\n";
    int pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7;
    switch (curType) {
        case LEX_IF:
            getLex();
            if (curType != LEX_LPAREN) {
                throw curLex;
            }
            getLex();
            E();
            exIsBool();
            if (curType != LEX_RPAREN) {
                throw curLex;
            }
            pl2 = prog.getFree();
            prog.blank();
            prog.putLex(Lex(POLIZ_FGO));
            getLex();
            O1();
            pl3 = prog.getFree();
            prog.blank();
            prog.putLex(Lex(POLIZ_GO));
            prog.putLex(Lex(POLIZ_LABEL, prog.getFree()), pl2);
            if (curType != LEX_ELSE) {
                throw curLex;
            }
            getLex();
            O1();
            prog.putLex(Lex(POLIZ_LABEL, prog.getFree()), pl3);
            break;

        case LEX_FOR:
            getLex();
            if (curType != LEX_LPAREN) {
                throw curLex;
            }
            getLex();
            E();
            if (curType != LEX_SEMICOLON) {
                throw curLex;
            }
            pl4 = prog.getFree();
            getLex();
            E();
            exIsBool();
            pl5 = prog.getFree();
            prog.blank();
            prog.putLex(Lex(POLIZ_FGO));
            if (curType != LEX_SEMICOLON) {
                throw curLex;
            }
            pl6 = prog.getFree();
            prog.blank();
            prog.putLex(Lex(POLIZ_GO));
            getLex();
            pl7 = prog.getFree();
            E();
            prog.putLex(Lex(POLIZ_LABEL, pl4));
            prog.putLex(Lex(POLIZ_GO));
            prog.putLex(Lex(POLIZ_LABEL, prog.getFree()), pl6);
            if (curType != LEX_RPAREN) {
                throw curLex;
            }
            getLex();
            deep++;
            O1();
            deep--;
            prog.putLex(Lex(POLIZ_LABEL, pl7));
            prog.putLex(Lex(POLIZ_GO));
            prog.putLex(Lex(POLIZ_LABEL, prog.getFree()), pl5);
            for(; breakCount > prevBreakCount; breakCount--) {
                prog.putLex(Lex(POLIZ_LABEL, prog.getFree()), breakStack.top());
                breakStack.pop();
            }
            break;

        case LEX_WHILE:
            prevBreakCount = breakCount;
            pl0 = prog.getFree();
            getLex();
            if (curType != LEX_LPAREN) {
                throw curLex;
            }
            getLex();
            E();
            exIsBool();
            if (curType != LEX_RPAREN) {
                throw curLex;
            }
            pl1 = prog.getFree();
            prog.blank();
            prog.putLex(Lex(POLIZ_FGO));
            getLex();
            deep++;
            O1();
            deep--;
            prog.putLex(Lex(POLIZ_LABEL, pl0));
            prog.putLex(Lex(POLIZ_GO));
            prog.putLex(Lex(POLIZ_LABEL, prog.getFree()), pl1);
            for(; breakCount > prevBreakCount; breakCount--) {
                prog.putLex(Lex(POLIZ_LABEL, prog.getFree()), breakStack.top());
                breakStack.pop();
            }
            break;

        case LEX_BREAK:
            if (deep == 0) {
                throw curLex;
            }
            breakStack.push(prog.getFree());
            prog.blank();
            prog.putLex(Lex(POLIZ_GO));
            breakCount++;
            getLex();
            if (curType != LEX_SEMICOLON) {
                throw curLex;
            }
            getLex();
            break;

        case LEX_READ:
            getLex();
            if (curType != LEX_LPAREN) {
                throw curLex;
            }
            getLex();
            idInRead = true;
            I();
            idInRead = false;
            if (idType == LEX_T_STRUCT) {
                throw curLex;
            }
            if (curType != LEX_RPAREN) {
                throw curLex;
            }
            prog.putLex(Lex(LEX_READ));
            getLex();
            if (curType != LEX_SEMICOLON) {
                throw curLex;
            }
            getLex();
            break;

        case LEX_WRITE:
            getLex();
            if (curType != LEX_LPAREN) {
                throw curLex;
            }
            getLex();
            E();
            prog.putLex(Lex(LEX_WRITE));
            if (opStack.top() == LEX_T_STRUCT) {
                throw curLex;
            }
            while (curType == LEX_COMMA) {
                getLex();
                E();
                prog.putLex(Lex(LEX_WRITE));
                if (opStack.top() == LEX_T_STRUCT) {
                    throw curLex;
                }
            }
            if (curType != LEX_RPAREN) {
                throw curLex;
            }
            getLex();
            if (curType != LEX_SEMICOLON) {
                throw curLex;
            }
            getLex();
            break;

        case LEX_LBRACE:
            getLex();
            while (curType != LEX_RBRACE) {
                O1();
            }
            getLex();
            break;

        case LEX_ID:
            if (labelMap.count(curStrLexValue) != 0) {
                (labelMap.find(curStrLexValue)->second).setPlace(prog.getFree());
                getLex();
                if (curType != LEX_COLON) {
                    throw curLex;
                }
                getLex();
                O1();
            } else {
                E();
                if (curType != LEX_SEMICOLON) {
                    throw curLex;
                }
                getLex();
            }
            break;

        case LEX_GOTO:
            getLex();
            if (curType != LEX_ID) {
                throw  curLex;
            }
            if (labelMap.count(curStrLexValue) == 0) {
                throw curLex;
            }
            tempLab.name = curLex.getStrLexValue();
            tempLab.place = prog.getFree();
            labelStack.push(tempLab);
            prog.blank();
            prog.putLex(Lex(POLIZ_GO));
            getLex();
            if (curType != LEX_SEMICOLON) {
                throw curLex;
            }
            getLex();
            break;

        default:
            throw curLex;
            break;
    }
}

void Parser::I() {
    cout << "I\n";
    if (curType != LEX_ID) {
        throw curLex;
    }
    tempStr1 = curStrLexValue;
    if (variablesMap.count(tempStr1) == 0) {
        throw curLex;
    }
    idType = (variablesMap.find(tempStr1)->second).getLexType();
    tempLex = curLex;
    getLex();
    if (curType != LEX_DOT) {
        if (idType == LEX_T_STRUCT) {
            prog.putLex(Lex(POLIZ_STRUCT_ID, 0, tempLex.getStrLexValue()));
            structStack.push((variablesMap.find(tempLex.getStrLexValue())->second).getStructName());
        } else {
            prog.putLex(Lex(POLIZ_VAR_ID, 0, tempLex.getStrLexValue()));
        }
        return;
    }
    getLex();
    if (curType != LEX_ID) {
        throw curLex;
    }
    tempStr2 = curStrLexValue;
    if ((variablesMap.find(tempStr1)->second).getStructFields().count(tempStr2) == 0) {
        throw curLex;
    }
    prog.putLex(Lex(POLIZ_STRUCT_FIELD_ID, 0, tempLex.getStrLexValue(), tempStr2));
    idType = ((variablesMap.find(tempStr1)->second).getStructFields().find(tempStr2)->second).getLexType();
    getLex();
}

void Parser::E() {
    cout << "E\n";
    onlyId = true;
    E1();
    if (curType == LEX_ASSIGN) {
        if (!onlyId) {
            throw curLex;
        }
        onlyId = true;
        getLex();
        E();
        tempType = opStack.top();
        opStack.pop();
        if (tempType != opStack.top()) {
            throw curLex;
        }
        if (tempType == LEX_T_STRUCT) {
            tempStr = structStack.top();
            structStack.pop();
            if (tempStr != structStack.top()) {
                throw curLex;
            }
        }
        prog.putLex(Lex(LEX_ASSIGN));
    }
}

void Parser::E1() {
    cout << "E1\n";
    E2();
    if (curType == LEX_OR) {
        onlyId = false;
        opStack.push(curType);
        getLex();
        E1();
        checkOp();
    }
}

void Parser::E2() {
    cout << "E2\n";
    E3();
    if (curType == LEX_AND) {
        onlyId = false;
        opStack.push(curType);
        getLex();
        E2();
        checkOp();
    }
}

void Parser::E3() {
    cout << "E3\n";
    E4();
    if ((curType == LEX_EQ) || (curType == LEX_LEQ) || (curType == LEX_GEQ)
            || (curType == LEX_LSS) || (curType == LEX_GTR) || (curType == LEX_NEQ)) {
        onlyId = false;
        opStack.push(curType);
        getLex();
        E4();
        checkOp();
    }
}

void Parser::E4() {
    cout << "E4\n";
    E5();
    if ((curType == LEX_PLUS) || (curType == LEX_MINUS)) {
        onlyId = false;
        opStack.push(curType);
        getLex();
        E4();
        checkOp();
    }
}

void Parser::E5() {
    cout << "E5\n";
    E6();
    if ((curType == LEX_TIMES) || (curType == LEX_SLASH)) {
        onlyId = false;
        opStack.push(curType);
        getLex();
        E5();
        checkOp();
    }
}

void Parser::E6() {
    cout << "E6\n";
    switch(curType) {
        case LEX_NOT:
            onlyId = false;
            getLex();
            E6();
            checkNot();
            break;

        case LEX_MINUS:
            onlyId = false;
            getLex();
            E6();
            checkUnMinus();
            break;

        case LEX_LPAREN:
            onlyId = false;
            getLex();
            E();
            if (curType != LEX_RPAREN) {
                throw curLex;
            }
            getLex();
            break;

        case LEX_NUM:
            onlyId = false;
            opStack.push(LEX_T_INT);
            prog.putLex(curLex);
            getLex();
            break;

        case LEX_STRING:
            onlyId = false;
            opStack.push(LEX_T_STRING);
            prog.putLex(curLex);
            getLex();
            break;

        case LEX_TRUE:
            onlyId = false;
            opStack.push(LEX_T_BOOLEAN);
            prog.putLex(curLex);
            getLex();
            break;

        case LEX_FALSE:
            onlyId = false;
            opStack.push(LEX_T_BOOLEAN);
            prog.putLex(curLex);
            getLex();
            break;

        case LEX_ID:
            I();
            opStack.push(idType);
            break;

        default:
            throw curLex;
            break;
    }
}









void Parser::execute() {
    cout << "@\n";
    Lex tempLex1, tempLex2;
    stack<Lex> args;
    int index = 0, i = 0, size = prog.getFree();

    while (index < size) {
        curLex = prog[index];
        //cout << curLex << endl;
        switch (curLex.getLexType()) {
            case LEX_TRUE:
            case LEX_FALSE:
            case LEX_NUM:
            case LEX_STRING:
            case POLIZ_VAR_ID:
            case POLIZ_STRUCT_FIELD_ID:
            case POLIZ_STRUCT_ID:
            case POLIZ_LABEL:
                args.push(curLex);
                break;

            case LEX_NOT: {
                tempLex1 = args.top();
                args.pop();
                switch (tempLex1.getLexType()) {
                    case LEX_TRUE:
                        args.push(Lex(LEX_FALSE, 0, "false"));
                        break;
                    case LEX_FALSE:
                        args.push(Lex(LEX_TRUE, 0, "true"));
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (!tempVar1.getBoolValue()) {
                            args.push(Lex(LEX_TRUE, 0, "true"));
                        } else {
                            args.push(Lex(LEX_FALSE, 0, "false"));
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (!tempVar1.getBoolValue()) {
                            args.push(Lex(LEX_TRUE, 0, "true"));
                        } else {
                            args.push(Lex(LEX_FALSE, 0, "false"));
                        }
                        break;
                    }
                }
                break;
            }

            case LEX_UN_MINUS: {
                tempLex1 = args.top();
                args.pop();
                switch (tempLex1.getLexType()) {
                    case LEX_NUM:
                        args.push(Lex(LEX_NUM, -tempLex1.getIntLexValue()));
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        args.push(Lex(LEX_NUM, -tempVar1.getIntValue()));
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        args.push(Lex(LEX_NUM, -tempVar1.getIntValue()));
                        break;
                    }
                }
                break;
            }

            case LEX_OR: {
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                bool b1, b2;

                switch (tempLex1.getLexType()) {
                    case LEX_TRUE:
                        b1 = true;
                        break;
                    case LEX_FALSE:
                        b1 = false;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b1 = tempVar1.getBoolValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b1 = tempVar1.getBoolValue();
                        break;
                    }
                }

                switch (tempLex2.getLexType()) {
                    case LEX_TRUE:
                        b2 = true;
                        break;
                    case LEX_FALSE:
                        b2 = false;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b2 = tempVar1.getBoolValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b2 = tempVar1.getBoolValue();
                        break;
                    }
                }

                b1 = b1 || b2;
                if (b1) {
                    args.push(Lex(LEX_TRUE, 0, "true"));
                } else {
                    args.push(Lex(LEX_FALSE, 0, "false"));
                }

                break;
            }

            case LEX_AND: {
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                bool b1, b2;

                switch (tempLex1.getLexType()) {
                    case LEX_TRUE:
                        b1 = true;
                        break;
                    case LEX_FALSE:
                        b1 = false;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b1 = tempVar1.getBoolValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b1 = tempVar1.getBoolValue();
                        break;
                    }
                }

                switch (tempLex2.getLexType()) {
                    case LEX_TRUE:
                        b2 = true;
                        break;
                    case LEX_FALSE:
                        b2 = false;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex2.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b2 = tempVar1.getBoolValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex2.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        b2 = tempVar1.getBoolValue();
                        break;
                    }
                }

                b1 = b1 && b2;
                if (b1) {
                    args.push(Lex(LEX_TRUE, 0, "true"));
                } else {
                    args.push(Lex(LEX_FALSE, 0, "false"));
                }

                break;
            }

            case POLIZ_GO: {
                index = args.top().getIntLexValue() - 1;
                args.pop();
                break;
            }

            case POLIZ_FGO: {
                i = args.top().getIntLexValue();
                args.pop();

                tempLex1 = args.top();
                args.pop();

                switch (tempLex1.getLexType()) {
                    case LEX_FALSE:
                        index = i - 1;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (tempVar1.getBoolValue()) {
                            index = i - 1;
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (tempVar1.getBoolValue()) {
                            index = i - 1;
                        }
                        break;
                    }
                }
                break;
            }

            case LEX_WRITE: {
                tempLex1 = args.top();
                args.pop();
                switch (tempLex1.getLexType()) {
                    case LEX_TRUE:
                    case LEX_FALSE:
                    case LEX_STRING:
                        cout << tempLex1.getStrLexValue() << endl;
                        break;
                    case LEX_NUM:
                        cout << tempLex1.getIntLexValue() << endl;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            cout << tempVar1.getIntValue() << endl;
                        } else if (tempVar1.getLexType() == LEX_T_BOOLEAN) {
                            if (tempVar1.getBoolValue()) {
                                cout << "true" << endl;
                            } else {
                                cout << "false" << endl;
                            }
                        } else {
                            cout << tempVar1.getStrValue() << endl;
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            cout << tempVar1.getIntValue() << endl;
                        } else if (tempVar1.getLexType() == LEX_T_BOOLEAN) {
                            if (tempVar1.getBoolValue()) {
                                cout << "true" << endl;
                            } else {
                                cout << "false" << endl;
                            }
                        } else {
                            cout << tempVar1.getStrValue() << endl;
                        }
                        break;
                    }
                }
                break;
            }

            case LEX_READ: {
                int k;
                tempLex1 = args.top();
                switch (tempLex1.getLexType()) {
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            cout << "Input int value for ";
                            cout << tempVar1.getName() << endl;
                            cin >> k;
                            tempVar1.setIntValue(k);
                            tempVar1.setAssign(true);
                        } else if (tempVar1.getLexType() == LEX_T_BOOLEAN) {
                            char j[20];
                            rep:
                            cout << "Input boolean value (true or false) for ";
                            cout << tempVar1.getName() << endl;
                            cin >> j;
                            if (!strcmp(j, "true")) {
                                tempVar1.setBoolValue(true);
                                tempVar1.setAssign(true);
                            } else if (!strcmp(j, "false")) {
                                tempVar1.setBoolValue(false);
                                tempVar1.setAssign(true);
                            } else {
                                cout << "Error in input:true/false" << endl;
                                goto rep;
                            }
                        } else {
                            char j[1000];
                            cout << "Input string for";
                            cout << tempVar1.getName() << endl;
                            cin >> j;
                            tempVar1.setStrValue(j);
                            tempVar1.setAssign(true);
                        }
                        break;
                    }

                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            cout << "Input int value for ";
                            cout << tempVar1.getName() << endl;
                            cin >> k;
                            tempVar1.setIntValue(k);
                            tempVar1.setAssign(true);
                        } else if (tempVar1.getLexType() == LEX_T_BOOLEAN) {
                            char j[20];
                            rep1:
                            cout << "Input boolean value (true or false) for ";
                            cout << tempVar1.getName() << endl;
                            cin >> j;
                            if (!strcmp(j, "true")) {
                                tempVar1.setBoolValue(true);
                                tempVar1.setAssign(true);
                            } else if (!strcmp(j, "false")) {
                                tempVar1.setBoolValue(false);
                                tempVar1.setAssign(true);
                            } else {
                                cout << "Error in input:true/false" << endl;
                                goto rep1;
                            }
                        } else {
                            char j[1000];
                            cout << "Input string for";
                            cout << tempVar1.getName() << endl;
                            cin >> j;
                            tempVar1.setStrValue(j);
                            tempVar1.setAssign(true);
                        }
                        break;
                    }
                }
                break;
            }

            case LEX_PLUS: {
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                int b1, b2, f;
                string c1, c2;

                switch (tempLex1.getLexType()) {
                    case LEX_NUM:
                        b1 = tempLex1.getIntLexValue();
                        f = 0;
                        break;
                    case LEX_STRING:
                        c1 = tempLex1.getStrLexValue();
                        f = 1;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            b1 = tempVar1.getIntValue();
                            f = 0;
                        } else {
                            c1 = tempVar1.getStrValue();
                            f = 1;
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            b1 = tempVar1.getIntValue();
                            f = 0;
                        } else {
                            c1 = tempVar1.getStrValue();
                            f = 1;
                        }
                        break;
                    }
                }

                switch (tempLex2.getLexType()) {
                    case LEX_NUM:
                        b2 = tempLex2.getIntLexValue();
                        f = 0;
                        break;
                    case LEX_STRING:
                        c2 = tempLex2.getStrLexValue();
                        f = 1;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar2 = variablesMap.find(tempLex2.getStrLexValue())->second;
                        if (!tempVar2.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar2.getLexType() == LEX_T_INT) {
                            b2 = tempVar2.getIntValue();
                            f = 0;
                        } else {
                            c2 = tempVar2.getStrValue();
                            f = 1;
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar2 = (variablesMap.find(tempLex2.getStrLexValue())->second)
                                .getStructFields().find(tempLex2.getStrLexValue1())->second;
                        if (!tempVar2.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar2.getLexType() == LEX_T_INT) {
                            b2 = tempVar2.getIntValue();
                            f = 0;
                        } else {
                            c2 = tempVar2.getStrValue();
                            f = 1;
                        }
                        break;
                    }
                }
                if (f == 0) {
                    args.push(Lex(LEX_NUM, b1 + b2));
                } else {
                    string str = c2 + c1;
                    char* c = new char[str.length() + 1];
                    for (int i = 0; i < str.length() + 1; i++) {
                        c[i] = str[i];
                    }
                    args.push(Lex(LEX_STRING, 0, c));
                }
                break;
            }

            case LEX_TIMES: {
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                int b1, b2;

                switch (tempLex1.getLexType()) {
                    case LEX_NUM:
                        b1 = tempLex1.getIntLexValue();
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        b1 = tempVar1.getIntValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        b1 = tempVar1.getIntValue();
                        break;
                    }
                }

                switch (tempLex2.getLexType()) {
                    case LEX_NUM:
                        b2 = tempLex1.getIntLexValue();
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex2.getStrLexValue())->second;
                        b2 = tempVar1.getIntValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex2.getStrLexValue())->second)
                                .getStructFields().find(tempLex2.getStrLexValue1())->second;
                        b2 = tempVar1.getIntValue();
                        break;
                    }
                }
                args.push(Lex(LEX_NUM, b1 * b2));
                break;
            }

            case LEX_MINUS: {
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                int b1, b2;

                switch (tempLex1.getLexType()) {
                    case LEX_NUM:
                        b1 = tempLex1.getIntLexValue();
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        b1 = tempVar1.getIntValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        b1 = tempVar1.getIntValue();
                        break;
                    }
                }

                switch (tempLex2.getLexType()) {
                    case LEX_NUM:
                        b2 = tempLex1.getIntLexValue();
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex2.getStrLexValue())->second;
                        b2 = tempVar1.getIntValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex2.getStrLexValue())->second)
                                .getStructFields().find(tempLex2.getStrLexValue1())->second;
                        b2 = tempVar1.getIntValue();
                        break;
                    }
                }
                args.push(Lex(LEX_NUM, b2 - b1));
                break;
            }

            case LEX_SLASH: {
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                int b1, b2;

                switch (tempLex1.getLexType()) {
                    case LEX_NUM:
                        b1 = tempLex1.getIntLexValue();
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        b1 = tempVar1.getIntValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        b1 = tempVar1.getIntValue();
                        break;
                    }
                }

                switch (tempLex2.getLexType()) {
                    case LEX_NUM:
                        b2 = tempLex1.getIntLexValue();
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex2.getStrLexValue())->second;
                        b2 = tempVar1.getIntValue();
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex2.getStrLexValue())->second)
                                .getStructFields().find(tempLex2.getStrLexValue1())->second;
                        b2 = tempVar1.getIntValue();
                        break;
                    }
                }
                if (b1 == 0) {
                    throw "Divide by zero!";
                }
                args.push(Lex(LEX_NUM, b2 / b1));
                break;
            }

            case LEX_EQ:
            case LEX_LSS:
            case LEX_GTR:
            case LEX_LEQ:
            case LEX_GEQ:
            case LEX_NEQ: {
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                int b1, b2, f;
                string c1, c2;

                switch (tempLex1.getLexType()) {
                    case LEX_NUM:
                        b1 = tempLex1.getIntLexValue();
                        f = 0;
                        break;
                    case LEX_STRING:
                        c1 = tempLex1.getStrLexValue();
                        f = 1;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            b1 = tempVar1.getIntValue();
                            f = 0;
                        } else {
                            c1 = tempVar1.getStrValue();
                            f = 1;
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                .getStructFields().find(tempLex1.getStrLexValue1())->second;
                        if (!tempVar1.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar1.getLexType() == LEX_T_INT) {
                            b1 = tempVar1.getIntValue();
                            f = 0;
                        } else {
                            c1 = tempVar1.getStrValue();
                            f = 1;
                        }
                        break;
                    }
                }

                switch (tempLex2.getLexType()) {
                    case LEX_NUM:
                        b2 = tempLex2.getIntLexValue();
                        f = 0;
                        break;
                    case LEX_STRING:
                        c2 = tempLex2.getStrLexValue();
                        f = 1;
                        break;
                    case POLIZ_VAR_ID: {
                        Variable &tempVar2 = variablesMap.find(tempLex2.getStrLexValue())->second;
                        if (!tempVar2.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar2.getLexType() == LEX_T_INT) {
                            b2 = tempVar2.getIntValue();
                            f = 0;
                        } else {
                            c2 = tempVar2.getStrValue();
                            f = 1;
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar2 = (variablesMap.find(tempLex2.getStrLexValue())->second)
                                .getStructFields().find(tempLex2.getStrLexValue1())->second;
                        if (!tempVar2.getAssign()) {
                            throw "Execution error: not assign!\n";
                        }
                        if (tempVar2.getLexType() == LEX_T_INT) {
                            b2 = tempVar2.getIntValue();
                            f = 0;
                        } else {
                            c2 = tempVar2.getStrValue();
                            f = 1;
                        }
                        break;
                    }

                }

                switch (curLex.getLexType()) {
                    case LEX_EQ:
                        if (f == 0) {
                            if (b1 == b2) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        } else {
                            if (c1 == c2) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        }
                        break;
                    case LEX_LSS:
                        if (f == 0) {
                            if (b2 < b1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        } else {
                            if (c2 < c1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        }
                        break;
                    case LEX_GTR:
                        if (f == 0) {
                            if (b2 > b1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        } else {
                            if (c2 > c1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        }
                        break;
                    case LEX_LEQ:
                        if (f == 0) {
                            if (b2 <= b1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        } else {
                            if (c2 <= c1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        }
                        break;
                    case LEX_GEQ:
                        if (f == 0) {
                            if (b2 >= b1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        } else {
                            if (c2 >= c1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        }
                        break;
                    case LEX_NEQ:
                        if (f == 0) {
                            if (b2 != b1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        } else {
                            if (c2 != c1) {
                                args.push(Lex(LEX_TRUE, 0, "true"));
                            } else {
                                args.push(Lex(LEX_FALSE, 0, "false"));
                            }
                        }
                        break;
                }

                break;
            }

            case LEX_ASSIGN:
                tempLex1 = args.top();
                args.pop();
                tempLex2 = args.top();
                args.pop();

                switch (tempLex2.getLexType()) {
                    case POLIZ_VAR_ID: {
                        Variable &tempVar = variablesMap.find(tempLex2.getStrLexValue())->second;
                        switch (tempLex1.getLexType()) {
                            case LEX_NUM:
                                tempVar.setIntValue(tempLex1.getIntLexValue());
                                tempVar.setAssign(true);
                                args.push(tempLex2);
                                break;
                            case LEX_STRING:
                                tempVar.setStrValue(tempLex1.getStrLexValue());
                                tempVar.setAssign(true);
                                args.push(tempLex2);
                                break;
                            case LEX_TRUE:
                                tempVar.setBoolValue(true);
                                tempVar.setAssign(true);
                                args.push(tempLex2);
                                break;
                            case LEX_FALSE:
                                tempVar.setBoolValue(false);
                                tempVar.setAssign(true);
                                args.push(tempLex2);
                                break;
                            case POLIZ_VAR_ID: {
                                Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                                tempVar = tempVar1;
                                tempVar.setName(tempLex2.getStrLexValue());
                                tempVar.setAssign(true);
                                args.push(tempLex2);
                                break;
                            }
                            case POLIZ_STRUCT_FIELD_ID: {
                                Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                        .getStructFields().find(tempLex1.getStrLexValue1())->second;
                                tempVar = tempVar1;
                                tempVar.setName(tempLex2.getStrLexValue());
                                tempVar.setAssign(true);
                                args.push(tempLex2);
                                break;
                            }
                        }
                        break;
                    }
                    case POLIZ_STRUCT_FIELD_ID: {
                        Variable &tempVar = (variablesMap.find(tempLex2.getStrLexValue())->second)
                                .getStructFields().find(tempLex2.getStrLexValue1())->second;
                        switch (tempLex1.getLexType()) {
                            case LEX_NUM:
                                tempVar.setIntValue(tempLex1.getIntLexValue());
                                tempVar.setAssign(true);
                                break;
                            case LEX_STRING:
                                tempVar.setStrValue(tempLex1.getStrLexValue());
                                tempVar.setAssign(true);
                                break;
                            case LEX_TRUE:
                                tempVar.setBoolValue(true);
                                tempVar.setAssign(true);
                                break;
                            case LEX_FALSE:
                                tempVar.setBoolValue(false);
                                tempVar.setAssign(true);
                                break;
                            case POLIZ_VAR_ID: {
                                Variable &tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                                tempVar = tempVar1;
                                tempVar.setName(tempLex2.getStrLexValue());
                                tempVar.setAssign(true);
                                break;
                            }
                            case POLIZ_STRUCT_FIELD_ID: {
                                Variable &tempVar1 = (variablesMap.find(tempLex1.getStrLexValue())->second)
                                        .getStructFields().find(tempLex1.getStrLexValue1())->second;
                                tempVar = tempVar1;
                                tempVar.setName(tempLex2.getStrLexValue());
                                tempVar.setAssign(true);
                                break;
                            }
                        }
                        args.push(tempLex2);
                        break;
                    }
                    case POLIZ_STRUCT_ID: {
                        Variable& tempVar = variablesMap.find(tempLex2.getStrLexValue())->second;
                        Variable& tempVar1 = variablesMap.find(tempLex1.getStrLexValue())->second;
                        tempVar.setStructFields(tempVar1.getStructFields());
                        break;
                    }
                }
                break;

            default:
                throw "POLIZ: unexpected elem";
        } // end of switch
        ++index;

    }; //end of while
    cout << "Finish of executing!!!" << endl;
}