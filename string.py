import json

class StringMethods:
    def __init__(self, input_string):
        self.input_string = input_string

    def spaceIndex(self):
        try:
            return self.input_string.index(' ')
        except ValueError:
            return -1

    def charCount(self, char):
        return len(char)

    def countWords(self):
        return len(self.input_string.split())

    def joinfirstLast(self):
        words = self.input_string.split()
        return '-'.join([words[0], words[-1]])
    
    def sortWordsChar(self):
    	words = self.input_string.split()
    	sorted_words = sorted(words, key=len, reverse=True)
    	return sorted_words


    
    def saveJson(self):
        save={
            "input":self.data,
            "space":self.spaceIndex,
            "count":self.charCount,
            "word":self.countWords,
            "join":self.joinfirstLast,     
        }
        with open('output.json','r+') as file:
            fiele_data=json.load(file)
            fiele_data['data'].append(save)
            file.seek(0)
            json.dump(fiele_data,file,indent=4)

string_methods = StringMethods("Hello World this is a test string")
spaceIndex = string_methods.spaceIndex()
print(f"the space index is at :{spaceIndex}")

charCounti = string_methods.charCount("i")
print(f"The total count of the character 'i' is: {charCounti}")

sortWORD = string_methods.sortWordsChar()
print(f"The sorted file is: {sortWORD}")

wordcount = string_methods.countWords()
print(f"The total count of words in the input string is: {wordcount}")

joinedWords = string_methods.joinfirstLast()
print(f"The first and last words of the input string joined with a hyphen are: {joinedWords}")
