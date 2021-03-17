'''
This file stores the definitions for functions and variables that are used more than once in the other python files.
'''
DIRECTORY_PATH = "D:\Third Year Project\CFIE-FRSE-master\output"

CSV_NAME = "output.csv"

OUTPUT_FILE_NAME = "random_forest.txt"

ARI_table = {
    5: "5th grade",
    6: "6th grade",
    7: "7th grade",
    8: "8th & 9th grade",
    9: "8th & 9th grade",
    10: "10th to 12th grade",
    11: "10th to 12th grade",
    12: "10th to 12th grade",
    13: "College",
    14: "Professional"
}

flesch_score_table = {}
for i in range(100):
    if i < 10:
        flesch_score_table[i] = "Professional"
    elif i < 30:
        flesch_score_table[i] = "College graduate"
    elif i < 50:
        flesch_score_table[i] = "College"
    elif i < 60:
        flesch_score_table[i] = "10th to 12th grade"
    elif i < 70:
        flesch_score_table[i] = "8th & 9th grade"
    elif i < 80:
        flesch_score_table[i] = "7th grade"
    elif i < 90:
        flesch_score_table[i] = "6th grade"
    elif i < 100:
        flesch_score_table[i] = "5th grade"

# This method is not used but it can be utilized inside the ModelTrain class if desired.
# Its purpose is to evaluate the baseline score
def baseline(self):
    # Only use the ARI scores that start from 5
    # If the ARI score is less than 5, then automatically count it out as a wrong
    print("Evaluating baseline score..")
    import math

    baseline_score = 0
    for i in range(len(self.y_test)):
        score = ""
        if self.ari_scores[i] > 14:
            score = "Professional"
        elif self.ari_scores[i] < 5:
            continue
        else:
            score = ARI_table[self.ari_scores[i][0]]

        if math.floor(self.y_test[i]) in self.flesch_score_table:
            if score == self.flesch_score_table[math.floor(self.y_test[i])]:
                baseline_score += 1
    
    print ("Baseline Scores")
    print(baseline_score)
    print("Length of ari_scores (can be used for rescaling to 1:100): " + str(len(self.ari_scores)))
    print("Length of y_test: " + str(len(self.y_test)))
    
    # print(baseline_score / 10)
    # print(baseline_score / 100)

    print("Length of y test: " + str(len(self.y_test)))

    print("ari_scores[i]: " + str(self.ari_scores[0]))
    print("ari_scores[i][j]: " + str(self.ari_scores[0][0]))

    return baseline_score
