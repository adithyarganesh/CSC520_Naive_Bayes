import csv, sys, os


# Splitting features and prediction
def create_dataset(df):
    features = df[0]
    df = df[1:]
    test_x, test_y = [], []
    for i in df:
        test_x.append(i[:-1])
        test_y.append(i[-1])
    return features, test_x, test_y


# Separating column names from data set
def load_csv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [x for x in dataset[i]]
    return create_dataset(dataset)


# Obtaining each class probability
def get_classes(features, trainx, trainy, mfile):
    modified_df, num_classes, class_object, model = {}, {}, {}, {}
    file = []
    for rowx, rowy in zip(trainx, trainy):
        if rowy not in modified_df.keys():
            modified_df[rowy] = [rowx]
        else:
            modified_df[rowy].append(rowx)
    for key, value in modified_df.items():
        num_classes[key] = len(modified_df[key])
        class_object[key] = num_classes[key] / len(trainy)
        model[key],temp = get_entity_probablity(features, key, value)
        file.extend(temp)
    modfile(file, mfile)
    return [list(num_classes.keys()), class_object, model]


# Determining entity probablity
def get_entity_probablity(features, key, dataset):
    model_sum = []
    temps = []
    datatranspose = zip(*dataset)
    for col, feature in zip(datatranspose, features):
        temp = list(set(col))
        elem_prob = {}
        # mfile.write("\nProbability of unique entities in " + str(feature) + "\n")
        for unique_prob in temp:
            elem_prob[unique_prob] = col.count(unique_prob)/len(dataset)
            s = "P(" + str(feature) + " = "+ str(unique_prob)+ "/Class = " +str(key) + "): " + str(elem_prob[unique_prob])
            temps.append(s)
            # mfile.write(unique_prob + ": " + str(elem_prob[unique_prob]) + "\n")
        model_sum.append(elem_prob)
    return model_sum, temps


def modfile(arr, m_file):
    try: os.remove(m_file)
    except: pass
    mfile = open(m_file, "a+")
    for i in arr:
            mfile.write(i + "\n")



# Prediction output with Nieve Bayes model
def get_model(model, test_x, test_y):
    num_classes = model[0]
    matrix = []
    for main_key, test_vals in enumerate(test_x):
        temp = []
        for entity in num_classes:
            probability = model[1][entity]
            for key, attribute  in enumerate(test_vals):
                try: probability *= model[2][entity][key][attribute]
                except: probability = 0
            temp.append(probability)
        matrix.append([num_classes[temp.index(max(temp))], test_y[main_key]])
    return matrix


# Confusion matrix
def confusion_matrix(matrix):
    return [[matrix.count(['1', '1']), matrix.count(['1', '0'])], [matrix.count(['0', '1']), matrix.count(['0', '0'])]]


# Writing predicted and expected output to a result file
def write_rrfile(matrix, mat, r_file):
    rfile = open(r_file, "a+")
    rfile.write("P: Predicted, E: Expected\n\nP\tE\n\n")
    for x,y in matrix:
        rfile.write(str(str(x) + "\t" + str(y) + "\n"))
    rfile.write("Confusion Matrix is: \n")
    for x,y in mat:
        rfile.write(str(str(x) + "\t" + str(y) + "\n"))
    rfile.close()


def main():
    train = sys.argv[1]
    test = sys.argv[2]
    m_file = sys.argv[3]
    r_file = sys.argv[4]
    try: os.remove(r_file)
    except: pass
    features, train_x,train_y = load_csv(train)
    _, test_x,test_y = load_csv(test)
    model = get_classes(features, train_x, train_y, m_file)
    matrix = get_model(model, test_x,test_y)
    mat = confusion_matrix(matrix)
    write_rrfile(matrix, mat, r_file)


if __name__ == '__main__':
    main()
