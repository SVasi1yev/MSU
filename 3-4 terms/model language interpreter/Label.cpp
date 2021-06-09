#include "Variable.cpp"
#include <string>

class Label {
    string name;
    int place;
public:
    Label (string name = "", int place = -1) {
        this->name = name;
        this->place = place;
    }

    string getName() {
        return name;
    }

    string setName (string name) {
        this->name = name;
    }

    int getPlace () {
        return place;
    }

    void setPlace (int place) {
        this->place = place;
    }

    friend ostream& operator << (ostream &s, Label v) {
        s << "(" << v.name << ", " << v.place << ");";
        return s;
    }
};