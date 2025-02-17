import wordninja as wn
import warnings

warnings.filterwarnings("ignore")

# Sets for operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
    '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';', '{', '}', '!', '~'
}

# keywords up to C11 and C++17; immutable set
keywords = frozenset({
    '__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export',
    '__far16', '__far32', '__fastcall', '__finally', '__import', '__inline',
    '__int16', '__int32', '__int64', '__int8', '__leave', '__optlink',
    '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
    '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except',
    '_Export', '_Far16', '_Far32', '_Fastcall', '_finally', '_Import',
    '_inline', '_int16', '_int32', '_int64', '_int8', '_leave', '_Optlink',
    '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas', 'alignof',
    'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
    'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const',
    'const_cast', 'constexpr', 'continue', 'decltype', 'default', 'delete',
    'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export',
    'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
    'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not',
    'not_eq', 'nullptr', 'operator', 'or', 'or_eq', 'override', 'private',
    'protected', 'public', 'register', 'reinterpret_cast', 'return', 'short',
    'signed', 'sizeof', 'static', 'static_assert', 'static_cast', 'struct',
    'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
    'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual',
    'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'
})
with open(r"sensiAPI.txt",
          "r") as f:
    a = f.read().split(',')
keywords = keywords.union(a)
# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})


def tokenize_code_line(line: str, subtoken: bool = False):
    """
    transform a string of code line into list of tokens

    Args:
        line: code line
        subtoken: whether to split into subtokens

    Returns:

    """
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 3].strip())
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 2].strip())
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []
    # Filter out irrelevant strings
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    # split subtoken
    res = list()
    if (subtoken):
        for token in tmp:
            res.extend(wn.split(token))
    else:
        res = tmp
    return res


if __name__ == '__main__':
    tokenize_code_line('youaresogreat hhh!', True)
