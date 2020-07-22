from sklearn import datasets
from sklearn.model_selection import train_test_split

# For using the mgbdt library, you have to include the library directory into your python path.
# If you are in this repository's root directory, you can do it by using the following lines
import sys
sys.path.insert(0, "lib")

from mGBDT.lib.mgbdt import MGBDT, MultiXGBModel

# make a sythetic circle dataset using sklearn
n_samples = 15000
x_all, y_all = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.04, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=0, stratify=y_all)

# Create a multi-layerd GBDTs
net = MGBDT(loss="CrossEntropyLoss", target_lr=1.0, epsilon=0.1)

# add several target-propogation layers
# F, G represent the forward mapping and inverse mapping (in this paper, we use gradient boosting decision tree)
net.add_layer("tp_layer",
    F=MultiXGBModel(input_size=2, output_size=5, learning_rate=0.1, max_depth=5, num_boost_round=5),
    G=None)
net.add_layer("tp_layer",
    F=MultiXGBModel(input_size=5, output_size=3, learning_rate=0.1, max_depth=5, num_boost_round=5),
    G=MultiXGBModel(input_size=3, output_size=5, learning_rate=0.1, max_depth=5, num_boost_round=5))
net.add_layer("tp_layer",
    F=MultiXGBModel(input_size=3, output_size=2, learning_rate=0.1, max_depth=5, num_boost_round=5),
    G=MultiXGBModel(input_size=2, output_size=3, learning_rate=0.1, max_depth=5, num_boost_round=5))

# init the forward mapping
net.init(x_train, n_rounds=5)

# fit the dataset
net.fit(x_train, y_train, n_epochs=50, eval_sets=[(x_test, y_test)], eval_metric="accuracy")

# prediction
y_pred = net.forward(x_test)

# get the hidden outputs
# hiddens[0] represent the input data
# hiddens[1] represent the output of the first layer
# hiddens[2] represent the output of the second layer
# hiddens[3] represent the output of the final layer (same as y_pred)
hiddens = net.get_hiddens(x_test)