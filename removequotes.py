

def removequotes():
    text = input("Text: ")
    for c in text:
        if c != "'":
            print(c, end="")
        

if __name__ == "__main__":
    removequotes()