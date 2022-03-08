#zad.1
import re
#a
text = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022"
result = re.sub(r'\d','' , text)
print(result)

#b
text = "<div><h2>Header</h2> <p>article<b>strong text</b><a href="">link</a></p></div>"
result = re.sub('<[^<]+?>','' , text)
print(result)

#c
text = """
Lorem ipsum dolor sit amet, consectetur; adipiscing elit.
Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros congue et. In
blandit, mi eu porta; lobortis, tortor nisl facilisis leo, at tristique augue risus
eu risus.
"""
result = re.sub(r'[^\w\s]','' , text)
print(result)

#zad.2
text = """
Lorem ipsum dolor
sit amet, consectetur adipiscing elit. Sed #texting eget mattis sem. Mauris #frasista
egestas erat #tweetext quam, ut faucibus eros #frasier congue et. In blandit, mi eu porta
lobortis, tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus.
"""
pattern = re.findall(r'#(\w+)', text)
print(pattern)

#zad.3
text = """
:)Lorem ipsum dolor;<
sit amet, c;)onsectetur adipiscing elit. Sed texting eget mattis sem. Mauris frasista
egestas erat tweetext quam, ut faucibus eros frasier :-)congue et. In blandit, mi eu porta
lobortis, ;( tortor nisl facilisis leo, at tristique frasistas augue risus eu risus.;-)
"""
pattern = re.findall(r'(:|;\W*)', text)
print(pattern)

#zad.4 na stronie

