

def removequotes():
    text = input("Text: ")
    for c in text:
        if c != "'" or c!= "\"":
            print(c, end="")
    
    [print() for i in range(4)]
        

if __name__ == "__main__":
    removequotes()