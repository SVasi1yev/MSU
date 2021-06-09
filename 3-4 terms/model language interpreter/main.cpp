#include <iostream>
#include "Parser.cpp"
int main() {
    Parser parser("program.txt");
    try {
        parser.analyze();
        parser.execute();
    } catch (Lex l) {
        cout << l << endl;
        parser.getScanner().showRowChar();
        parser.printLabels();
        parser.printVariables();
        parser.printStructs();
        //parser.printPoliz();
    } catch (const char* c) {
        cout << c << endl;
        parser.printLabels();
        parser.printVariables();
        parser.printStructs();
        //parser.printPoliz();
    }
    parser.printLabels();
    parser.printVariables();
    parser.printStructs();
    //parser.printPoliz();
}