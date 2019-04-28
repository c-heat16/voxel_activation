import pickle
#import cPickle as pickle
#func = pickle.load(open("common_voxel_col_to_coord.pkl", "wb"))
#print(func)


pickle_in = open("common_voxel_col_to_coord.pkl", "rb")
example_dict = pickle.load(pickle_in)

#with open("common_voxel_col_to_coord.pkl", "r") as f:
#    data = pickle.load(f)

#for filename in "common_voxel_col_to_coord.pkl":


try:
    print(2)
    with open("common_voxel_col_to_coord.pkl", 'rb') as f:
        print(1)
        data = pickle.load(f)
        # use the data
        except EOFError:
            continue