from final import model
import numpy as np

name = input("Enter name of appllicant: ")
print("\nThe scores of all the tests in quiz as well as survey need to be entered.")
print("All the values lie in the range 0 to 1.\n")
lang_vocab = float(input("Enter the score of Language Vocab test: "))
memory = float(input("Enter the score of Memory test: "))
speed = float(input("Enter the score of Speed test: "))
visual = float(input("Enter the score of Visual Discrimination test: "))
audio = float(input("Enter the score of Audio Discrimination test: "))
survey = float(input("Enter the score obtained from Survey: "))
def get_result(lang_vocab, memory, speed, visual, audio, survey):
    array = np.array([[lang_vocab, memory, speed, visual, audio, survey]])
    label = int(model.predict(array))
    if(label == 0):
        output = "There is a high chance of the applicant to have dyslexia."
    elif(label == 1):
        output = "There is a moderate chance of the applicant to have dyslexia."
    else:
        output = "There is a low chance of the applicant to have dyslexia."
    return output
get_result(lang_vocab, memory, speed, visual, audio, survey)
print(get_result(lang_vocab, memory, speed, visual, audio, survey))