import cmd
import shlex
import inspect

import pynames


class NameGenerator(cmd.Cmd):
    intro = 'Name generator'
    prompt = '>>> '

    langs = {
        'ru': pynames.LANGUAGE.RU,
        'en': pynames.LANGUAGE.EN,
        'native': pynames.LANGUAGE.NATIVE
    }
    genders = {
        'male': pynames.GENDER.MALE,
        'female': pynames.GENDER.FEMALE
    }

    def __init__(self):
        super().__init__()
        self.lang = 'native'

        suffixes = ['NamesGenerator', 'FullnameGenerator', 'Generator']
        self.classes = {}
        for class_ in pynames.generators.__all__:
            self.classes[class_.lower()] = {}
            for mem in inspect.getmembers(getattr(pynames.generators, class_)):
                for suffix in suffixes:
                    if mem[0].endswith(suffix) and not mem[0].startswith('From'):
                        self.classes[class_][mem[0][:-len(suffix)].lower()] = mem[1]
                        if '__default__' not in self.classes[class_]:
                            self.classes[class_]['__default__'] = mem[1]
                        break

    def parse(self, arg):
        args = shlex.split(arg)
        class_ = args[0].lower()
        subclass = self.classes[class_]['__default__']
        gender = self.genders['male']
        include_gender = False
        if len(args) == 2:
            if args[1].lower() in self.genders:
                gender = self.genders[args[1].lower()]
                include_gender = True
            else:
                subclass = self.classes[class_][args[1].lower()]
        if len(args) == 3:
            subclass = self.classes[class_][args[1].lower()]
            gender = self.genders[args[2].lower()]
            include_gender = True

        return class_, subclass, gender, include_gender

    def do_generate(self, arg):
        class_, subclass, gender, _ = self.parse(arg)
        lang = self.lang
        if lang not in subclass().languages:
            lang = 'native'
        print(class_, subclass, gender, lang)
        name = subclass().get_name_simple(gender, self.langs[lang])
        print(name)

    def complete_generate(self, prefix, line, begidx, endidx):
        last = line.split(' ')[-2]
        if last == 'generate':
            words = list(self.classes.keys())
        elif last in self.classes and len(self.classes[last]) > 1:
            words = (list(self.classes[last].keys()) + list(self.genders.keys()))
            words.remove('__default__')
        else:
            words = list(self.genders.keys())

        words = [word for word in words if word.lower().startswith(prefix.lower())]
        return words

    def do_language(self, arg):
        lang, *_ = shlex.split(arg)
        if lang.lower() in self.langs:
            self.lang = lang.lower()
        else:
            print('wrong language')

    def complete_language(self, prefix, line, begidx, endidx):
        return [word for word in self.langs if word.startswith(prefix.lower())]

    def do_info(self, arg):
        args = shlex.split(arg)
        if args[-1] == 'language':
            class_, subclass, _, _ = self.parse(' '.join(args[:-1]))
            print(*subclass().languages)
        else:
            class_, subclass, gender, include_gender = self.parse(arg)
            if include_gender:
                print(subclass().get_names_number(gender))
            else:
                print(subclass().get_names_number())

    def complete_info(self, prefix, line, begidx, endidx):
        last = line.split(' ')[-2]
        if last == 'info':
            words = list(self.classes.keys())
        elif last in self.classes and len(self.classes[last]) > 1:
            words = (list(self.classes[last].keys()) + list(self.genders.keys())) + ['language']
            words.remove('__default__')
        else:
            words = list(self.genders.keys()) + ['language']

        words = [word for word in words if word.lower().startswith(prefix.lower())]
        return words


if __name__ == '__main__':
    generator = NameGenerator()
    generator.cmdloop()
