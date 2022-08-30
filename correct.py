from hanspell_mod.hanspell import spell_checker
import pprint


text = "은서네 집에서 초코 케이크를 만들엇따. 친구한테 뭐 하냐고 전화했는데 집에와서 케이크를 만들자고 했기때문이다. 초코를 엄청 많이 너었고 생크림도 듬뿍 발랐다. 만들기는 정말 쉬웠는데 정말 정말 맛있엇따! 곧 생일인 인경이에게 선물해줘야겠다. 맛있겠다!"

def correct(text):
    result = spell_checker.check(text)
    result = result.as_dict()
    checked = result['checked']
    original = result['original']
    words = result['words']

    error_list = [k for k, v in words.items() if v != 0]

    corrected = []
    new = checked

    for error in error_list:
        try:
            before, new = new.split(error)
            corrected.append(before)
            corrected.append("#"+error)
        except:
            break
    corrected.append(new)
    
    correct_string = checked

    return original, corrected, correct_string


pprint.pprint(correct(text))