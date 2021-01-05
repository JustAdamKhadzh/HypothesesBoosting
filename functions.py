import pandas as pd

def transform_to_description(data):
    transformed_data = pd.DataFrame(columns=data.columns)
    
    for col in data:
        transformed_data[col] = data[col].apply(lambda x: (x, x))
    
    return transformed_data

def similarity(vect1, vect2):
    
    """
    previous version:
     vect1 = transformed_train.iloc[0]
     vect2 = transformed_train.iloc[2]
     func = (lambda x,y: (min(x[0], y[0]), max(x[1], y[1])))
     pd.Series(map(func, vect1, vect2), index=train_data.columns)
    
     for col in cols:
     vect_min = min(vect1.loc[col][0], vect2.loc[col][0])
     vect_max = max(vect1.loc[col][1], vect2.loc[col][1])
     vect[col] = (vect_min, vect_max)
    """
    
    func = lambda x, y: (min(x[0], y[0]), max(x[1], y[1]))
    vect = pd.Series(map(func, vect1, vect2), index=vect1.index)
    return vect

def inclusion(obj, patterns):
    """
    check where an obj is inluded in list of patterns
    is_include = any([all(obj == elem) for elem in patterns])
    """     
    is_include = any([obj.equals(elem) for elem in patterns])
    return  is_include

def get_similarity_sample_repr(sample: pd.DataFrame):
    
    """
    get sample of feature represantations from pos or neg class dataset
    returns feature represantation for sample by similarity operation
    """
    pattern = None
    for i, obj in sample.iterrows():
        if pattern is None:
            pattern = obj
        pattern = similarity(pattern, obj)
    return pattern
    
#операция нахождения объектов по признаковому представлению
def is_included_in_repr(d, train_data):
    """
    returns objects from train dataset(from train pos and neg data) that is included in d representation
    """
    
    d_list = []
    
    for i, obj in train_data.iterrows():
        feature_repr = similarity(obj, d)
        is_included = d.equals(feature_repr)
        if is_included:
            d_list.append(obj)
            
    return d_list