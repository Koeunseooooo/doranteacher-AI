from hanspell_mod.hanspell import spell_checker
import pprint


text = '''은서네 집에가서 초코케익을 만들었다. 
친구한테 뭐하냐고 전화했는데 집에 와서 케이크를 만들자고 했기때문이다. 
케익에 생크림이랑 초콜릿을 넣었다. 체리도 엄청 올렸다. 
만들기는 정말 쉬웠는데 정말 정말 맛있었다! 
곳 생일인 인경이에게 선물해 줘야겠다. 맛있겠다!'''


result = spell_checker.check(text)
pprint.pprint(result.as_dict())

# def correct(text):
