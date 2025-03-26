import random

with open("test.csv",'r') as f:
    data = [x.strip().split(",") for x in f]
f.close()

feature_names = data[0]
data = data[1:]

def random_forest(n):
    models = []
    for i in range(n):
        chosen_features = random.sample(feature_names[:len(feature_names) - 1], 3) # 3 CAN BE SQRTN - LOG N
        chosen_features.append(feature_names[len(feature_names) - 1])
        indices = [feature_names.index(q) for q in chosen_features]
        indices.append(len(feature_names) - 1)

        new_data = []

        for i in data:
            new_data.append([i[x] for x in range(len(i)) if x in indices])
        
        new_data = random.choices(new_data, k=3) # K DEPENDS ON DATASET HERE
        print(new_data)


    return models

random_forest(3)