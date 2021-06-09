#include <cstring>
#include <iostream>
#include <string>
#include "LexScanner.cpp"

using namespace std;

class Variable {
    string name;
    LexType type;
    int intValue;
    bool boolValue;
    string strValue;
    string structName;
    map <string, Variable> structFields;
    //bool declare;
    bool assign;
public:
    Variable (string name = "", LexType type = LEX_NULL, int intValue = 0,
            string strValue = "", string structName = "", map <string, Variable> structFields = map <string, Variable>(), bool assign = false) {
        this->name = name;
        this->type = type;
        this->intValue = intValue;
        this->strValue = strValue;
        this->structName = structName;
        this->structFields = structFields;
        this->assign = assign;
        if (intValue == 0) {
            boolValue = false;
        } else {
            boolValue = true;
        }
    }

    string getName () {
        return name;
    }

    void setName (string name) {
        this->name = name;
    }

    LexType getLexType () {
        return type;
    }

    void setLexType (LexType type) {
        this->type = type;
    }

    int getIntValue () {
        return intValue;
    }

    void setIntValue (int intValue) {
        this->intValue = intValue;
        if (intValue == 0) {
            boolValue = false;
        } else {
            boolValue = true;
        }
    }

    bool getBoolValue () {
        return boolValue;
    }

    void setBoolValue (bool boolValue) {
        this->boolValue = boolValue;
        if (boolValue) {
            intValue = 1;
        } else {
            intValue = 0;
        }
    }

    string getStrValue () {
        return strValue;
    }

    void setStrValue (string strValue) {
        this->strValue = strValue;
    }

    bool getAssign () {
        return assign;
    }

    void setAssign (bool assign) {
        this->assign = assign;
    }

    string getStructName () {
        return structName;
    }

    map <string, Variable>& getStructFields () {
        return structFields;
    };

    void setStructFields (map <string, Variable> a) {
        structFields = a;
    }

    friend ostream& operator << (ostream &s, Variable v) {
        s << "(" << v.name
          << ", " << Lex::lexTypeToString(v.type)
          << ", " << v.intValue
          << ", " << v.boolValue
          << ", " << '"' << v.strValue
          << "\", " << v.structName << ", " << v.assign << ");";
        return s;
    }
};