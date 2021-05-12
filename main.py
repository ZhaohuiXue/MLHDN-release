# main script
"""
@author: mengxue.zhang
"""


from model import get_model, cal_input_shape
from train_test_script import run_model, test_images
import time


mode = 'EVAL' # 'DEBUG' mode prints the training info, 'EVAL' mode prints none.
train_prefix = 'train_10' # training meta-file prefix
test_prefix = 'test' # test meta-file prefix
run_times = 10 # run times
epochs = 10 # epochs
lr = 1e-3 # learning rate
d = [1, 3, 9, 27] # dilation rate array
batch_size = 32 # batch size


# training and test model
def soc_model(base_dir='./soe/MLHDN', train_prefix='train_10', test_prefix='test', name='MLHDN', image_shape=[56, 56], hyperparams={'r1': 4, 'r2': 1, 't': 'mlrbp', 'v': 4, 'd': [1, 3, 7, 15]},
              epochs=10):
    model = get_model(classes=10, lr=lr, name=name, image_shape=image_shape, hyperparams=hyperparams)
    run_model(model, mode, base_dir, train_prefix, test_prefix, run_times, image_shape,
              epochs=epochs, batch_size=batch_size, lr=lr)

# test without training
def soc_test(base_dir='./soe', model_path='./mv_10', test_prefix='test', name='MLHDN',
             image_shape=[56, 56], hyperparams={'r1': 4, 'r2': 1, 't': 'mlrbp', 'v': 4, 'd': [1, 3, 7, 15]}, suffix=None):
    model = get_model(classes=10, lr=lr, name=name, image_shape=image_shape, hyperparams=hyperparams)
    test_images(model, mode, model_path, base_dir, test_prefix, run_times, image_shape, suffix=suffix)


# classification experiments
shp = cal_input_shape(d)
soc_model(base_dir='./soe/MLHDN', name='MLHDN', image_shape=shp, train_prefix=train_prefix, test_prefix=test_prefix,
          hyperparams={'r1': 4, 'r2': 2, 't': 'mlrbp', 'v': 4, 'd': d}, epochs=epochs)

## noise experiments
# shp = cal_input_shape(d)
# for i in range(1, 16):
#     soc_test(base_dir='./noise/MLHDN', model_path='./soe/MLHDN', test_prefix='test', name='MLHDN',
#              image_shape=shp, hyperparams={'r1': 4, 'r2': 2, 't': 'mlrbp', 'v': 4, 'd': d}, suffix='n'+str(i))

## theta estimation error experiments
# shp = cal_input_shape(d)
# for i in [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
#     soc_test(base_dir='./theta/MLHDN', model_path='./soe/MLHDN', test_prefix='test', name='MLHDN',
#              image_shape=shp, hyperparams={'r1': 4, 'r2': 2, 't': 'mlrbp', 'v': 4, 'd': d}, suffix='t'+str(i))

## view experiments
# shp = cal_input_shape(d)
# for v in range(2, 10):
#     soc_model(base_dir='./mv_' + str(v) + '/MLHDN', name='MLHDN', image_shape=shp, train_prefix=train_prefix,
#                test_prefix=test_prefix, hyperparams={'r1': 4, 'r2': 2, 't': 'mlrbp', 'v': v, 'd': d}, epochs=epochs)
# for v in range(2, 10):
#     soc_test(base_dir='./view_'+ str(v) + '/MLHDN', model_path='./mv_' + str(v) + '/MLHDN', test_prefix='test', name='MLHDN',
#                  image_shape=shp, hyperparams={'r1': 4, 'r2': 2, 't': 'mlrbp', 'v': v, 'd': d}, suffix='v'+str(v))



