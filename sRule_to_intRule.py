x = 5

def string_rule_to_int_list(input_rule):
    new_rule = []
    for char in input_rule:
        if (char == '1'):
            new_rule.append(1)
        else: 
            new_rule.append(0)
    print(x)
    return new_rule

if __name__ == "__main__":
    rule = "101"
    x=10
    print(string_rule_to_int_list(rule))