# Preprocessing with scikit-learn - Cumulative Lab

## Introduction
In this cumulative lab, you'll practice applying various preprocessing techniques with scikit-learn (`sklearn`) to the Ames Housing dataset in order to prepare the data for predictive modeling. The main emphasis here is on preprocessing (not EDA or modeling theory), so we will skip over most of the visualization and metrics steps that you would take in an actual modeling process.

## Objectives

You will be able to:

* Practice identifying which preprocessing technique to use
* Practice filtering down to relevant columns
* Practice applying `sklearn.impute` to fill in missing values
* Practice applying `sklearn.preprocessing`:
  * `LabelBinarizer` for converting binary categories to 0 and 1 within a single column
  * `OneHotEncoder` for creating multiple "dummy" columns to represent multiple categories
  * `PolynomialFeatures` for creating interaction terms
  * `StandardScaler` for scaling data

## Your Task: Prepare the Ames Housing Dataset for Modeling

![house in Ames](images/ames_house.jpg)

<span>Photo by <a href="https://unsplash.com/@kjkempt17?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Kyle Kempt</a> on <a href="https://unsplash.com/s/photos/ames?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

### Requirements

#### 1. Drop Irrelevant Columns

For the purposes of this lab, we will only be using a subset of all of the features present in the Ames Housing dataset. In this step you will drop all irrelevant columns.

#### 2. Handle Missing Values

Often for reasons outside of a data scientist's control, datasets are missing some values. In this step you will assess the presence of NaN values in our subset of data, and use `MissingIndicator` and `SimpleImputer` from the `sklearn.impute` submodule to handle any missing values.

#### 3. Convert Categorical Features into Numbers

A built-in assumption of the scikit-learn library is that all data being fed into a machine learning model is already in a numeric format, otherwise you will get a `ValueError` when you try to fit a model. In this step you will use a `LabelBinarizer` to replace data within individual non-numeric columns with 0s and 1s, and a `OneHotEncoder` to replace columns containing more than 2 categories with multiple "dummy" columns containing 0s and 1s.

At this point, a scikit-learn model should be able to run without errors!

#### 4. Add Interaction Terms

This step gets into the feature engineering part of preprocessing. Does our model improve as we add interaction terms? In this step you will use a `PolynomialFeatures` transformer to augment the existing features of the dataset.

#### 5. Scale Data

Because we are using a model with regularization, it's important to scale the data so that coefficients are not artificially penalized based on the units of the original feature. In this step you will use a `StandardScaler` to standardize the units of your data.

#### BONUS: Refactor into a Pipeline

In a professional data science setting, this work would be accomplished mainly within a scikit-learn pipeline, not by repeatedly creating pandas `DataFrame`s, transforming them, and concatenating them together. In this step you will optionally practice refactoring your existing code into a pipeline (or you can just look at the solution branch).

## Lab Setup

### Getting the Data

In the cell below, we import the `pandas` library, open the CSV containing the Ames Housing data as a pandas `DataFrame`, and inspect its contents.


```python
# Run this cell without changes
import pandas as pd
df = pd.read_csv("data/ames.csv")
df
```


```python
# __SOLUTION__
import pandas as pd
df = pd.read_csv("data/ames.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1456</td>
      <td>60</td>
      <td>RL</td>
      <td>62.0</td>
      <td>7917</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>20</td>
      <td>RL</td>
      <td>85.0</td>
      <td>13175</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>RL</td>
      <td>66.0</td>
      <td>9042</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>2500</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>9717</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 81 columns</p>
</div>




```python
# Run this cell without changes
df.describe()
```


```python
# __SOLUTION__
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>



The prediction target for this analysis is the sale price of the home, so we separate the data into `X` and `y` accordingly:


```python
# Run this cell without changes
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)
```


```python
# __SOLUTION__
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)
```

Next, we separate the data into a train set and a test set prior to performing any preprocessing steps:


```python
# Run this cell without changes
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
# __SOLUTION__
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

(If you are working through this lab and you just want to start over with the original value for `X_train`, re-run the cell above.)


```python
# Run this cell without changes
print(f"X_train is a DataFrame with {X_train.shape[0]} rows and {X_train.shape[1]} columns")
print(f"y_train is a Series with {y_train.shape[0]} values")

# We always should have the same number of rows in X as values in y
assert X_train.shape[0] == y_train.shape[0]
```


```python
# __SOLUTION__
print(f"X_train is a DataFrame with {X_train.shape[0]} rows and {X_train.shape[1]} columns")
print(f"y_train is a Series with {y_train.shape[0]} values")

# We always should have the same number of rows in X as values in y
assert X_train.shape[0] == y_train.shape[0]
```

    X_train is a DataFrame with 1095 rows and 80 columns
    y_train is a Series with 1095 values


#### Fitting a Model

For this lab we will be using an `ElasticNet` model from scikit-learn ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)). You are welcome to read about the details of this model implementation at that link, but for the purposes of this lab, what you need to know is that this is a form of linear regression with *regularization* (meaning we will need to standardize the features).

Right now, we have not done any preprocessing, so we expect that trying to fit a model will fail:


```python
# Run this cell without changes
from sklearn.linear_model import ElasticNet

model = ElasticNet(random_state=1)
model.fit(X_train, y_train)
```


```python
# __SOLUTION__
from sklearn.linear_model import ElasticNet

model = ElasticNet(random_state=1)
model.fit(X_train, y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-6-6102dd698287> in <module>
          3 
          4 model = ElasticNet(random_state=1)
    ----> 5 model.fit(X_train, y_train)
    

    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py in fit(self, X, y, sample_weight, check_input)
        757         if check_input:
        758             X_copied = self.copy_X and self.fit_intercept
    --> 759             X, y = self._validate_data(X, y, accept_sparse='csc',
        760                                        order='F',
        761                                        dtype=[np.float64, np.float32],


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        430                 y = check_array(y, **check_y_params)
        431             else:
    --> 432                 X, y = check_X_y(X, y, **check_params)
        433             out = X, y
        434 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
        793         raise ValueError("y cannot be None")
        794 
    --> 795     X = check_array(X, accept_sparse=accept_sparse,
        796                     accept_large_sparse=accept_large_sparse,
        797                     dtype=dtype, order=order, copy=copy,


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         70                           FutureWarning)
         71         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 72         return f(**kwargs)
         73     return inner_f
         74 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        596                     array = array.astype(dtype, casting="unsafe", copy=False)
        597                 else:
    --> 598                     array = np.asarray(array, order=order, dtype=dtype)
        599             except ComplexWarning:
        600                 raise ValueError("Complex data not supported\n"


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/pandas/core/generic.py in __array__(self, dtype)
       1779 
       1780     def __array__(self, dtype=None) -> np.ndarray:
    -> 1781         return np.asarray(self._values, dtype=dtype)
       1782 
       1783     def __array_wrap__(self, result, context=None):


    //anaconda3/envs/learn-env/lib/python3.8/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 


    ValueError: could not convert string to float: 'RL'


As you can see, we got `ValueError: could not convert string to float: 'RL'`.

In order to fit a scikit-learn model, all values must be numeric, and the third column of our full dataset (`MSZoning`) contains values like `'RL'` and `'RH'`, which are strings. So this error was expected, but after some preprocessing, this model will work!

## 1. Drop Irrelevant Columns

For the purpose of this analysis, we'll only use the following columns, described by `relevant_columns`. You can find the full description of their values in the file `data/data_description.txt` included in this repository.

In the cell below, reassign `X_train` so that it only contains the columns in `relevant_columns`.

**Hint:** Even though we describe this as "dropping" irrelevant columns, it's easier if you invert the logic, so that we are only keeping relevant columns, rather than using the `.drop()` method. It is possible to use the `.drop()` method if you really want to, but first you would need to create a list of the column names that you don't want to keep.


```python
# Replace None with appropriate code

# Declare relevant columns
relevant_columns = [
    'LotFrontage',  # Linear feet of street connected to property
    'LotArea',      # Lot size in square feet
    'Street',       # Type of road access to property
    'OverallQual',  # Rates the overall material and finish of the house
    'OverallCond',  # Rates the overall condition of the house
    'YearBuilt',    # Original construction date
    'YearRemodAdd', # Remodel date (same as construction date if no remodeling or additions)
    'GrLivArea',    # Above grade (ground) living area square feet
    'FullBath',     # Full bathrooms above grade
    'BedroomAbvGr', # Bedrooms above grade (does NOT include basement bedrooms)
    'TotRmsAbvGrd', # Total rooms above grade (does not include bathrooms)
    'Fireplaces',   # Number of fireplaces
    'FireplaceQu',  # Fireplace quality
    'MoSold',       # Month Sold (MM)
    'YrSold'        # Year Sold (YYYY)
]

# Reassign X_train so that it only contains relevant columns
None

# Visually inspect X_train
X_train
```


```python
# __SOLUTION__

# Declare relevant columns
relevant_columns = [
    'LotFrontage',  # Linear feet of street connected to property
    'LotArea',      # Lot size in square feet
    'Street',       # Type of road access to property
    'OverallQual',  # Rates the overall material and finish of the house
    'OverallCond',  # Rates the overall condition of the house
    'YearBuilt',    # Original construction date
    'YearRemodAdd', # Remodel date (same as construction date if no remodeling or additions)
    'GrLivArea',    # Above grade (ground) living area square feet
    'FullBath',     # Full bathrooms above grade
    'BedroomAbvGr', # Bedrooms above grade (does NOT include basement bedrooms)
    'TotRmsAbvGrd', # Total rooms above grade (does not include bathrooms)
    'Fireplaces',   # Number of fireplaces
    'FireplaceQu',  # Fireplace quality
    'MoSold',       # Month Sold (MM)
    'YrSold'        # Year Sold (YYYY)
]

# Reassign X_train so that it only contains relevant columns
X_train = X_train.loc[:, relevant_columns]


# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>626</th>
      <td>NaN</td>
      <td>12342</td>
      <td>Pave</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>Pave</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>Pave</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



Check that the new shape is correct:


```python
# Run this cell without changes

# X_train should have the same number of rows as before
assert X_train.shape[0] == 1095

# Now X_train should only have as many columns as relevant_columns
assert X_train.shape[1] == len(relevant_columns)
```


```python
# __SOLUTION__

# X_train should have the same number of rows as before
assert X_train.shape[0] == 1095

# Now X_train should only have as many columns as relevant_columns
assert X_train.shape[1] == len(relevant_columns)
```

## 2. Handle Missing Values

In the cell below, we check to see if there are any NaNs in the selected subset of data:


```python
# Run this cell without changes
X_train.isna().sum()
```


```python
# __SOLUTION__
X_train.isna().sum()
```




    LotFrontage     200
    LotArea           0
    Street            0
    OverallQual       0
    OverallCond       0
    YearBuilt         0
    YearRemodAdd      0
    GrLivArea         0
    FullBath          0
    BedroomAbvGr      0
    TotRmsAbvGrd      0
    Fireplaces        0
    FireplaceQu     512
    MoSold            0
    YrSold            0
    dtype: int64



Ok, it looks like we have some NaNs in `LotFrontage` and `FireplaceQu`.

Before we proceed to fill in those values, we need to ask: **do these NaNs actually represent** ***missing*** **values, or is there some real value/category being represented by NaN?**

### Fireplace Quality

To start with, let's look at `FireplaceQu`, which means "Fireplace Quality". Why might we have NaN fireplace quality?

Well, some properties don't have fireplaces!

Let's confirm this guess with a little more analysis.

First, we know that there are 512 records with NaN fireplace quality. How many records are there with zero fireplaces?


```python
# Run this cell without changes
X_train[X_train["Fireplaces"] == 0]
```


```python
# __SOLUTION__
X_train[X_train["Fireplaces"] == 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>839</th>
      <td>70.0</td>
      <td>11767</td>
      <td>Pave</td>
      <td>5</td>
      <td>6</td>
      <td>1946</td>
      <td>1995</td>
      <td>1200</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>430</th>
      <td>21.0</td>
      <td>1680</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1971</td>
      <td>1971</td>
      <td>987</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>513</th>
      <td>71.0</td>
      <td>9187</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1983</td>
      <td>1983</td>
      <td>1080</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>40.0</td>
      <td>3951</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2009</td>
      <td>2009</td>
      <td>1224</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>330</th>
      <td>NaN</td>
      <td>10624</td>
      <td>Pave</td>
      <td>5</td>
      <td>4</td>
      <td>1964</td>
      <td>1964</td>
      <td>1728</td>
      <td>2</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>NaN</td>
      <td>11</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>63.0</td>
      <td>13072</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>1141</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>121</th>
      <td>50.0</td>
      <td>6060</td>
      <td>Pave</td>
      <td>4</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1123</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
<p>512 rows × 15 columns</p>
</div>



Ok, that's 512 rows, same as the number of NaN `FireplaceQu` records. To double-check, let's query for that combination of factors (zero fireplaces and `FireplaceQu` is NaN):


```python
# Run this cell without changes
X_train[
    (X_train["Fireplaces"] == 0) &
    (X_train["FireplaceQu"].isna())
]
```


```python
# __SOLUTION__
X_train[
    (X_train["Fireplaces"] == 0) &
    (X_train["FireplaceQu"].isna())
]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>839</th>
      <td>70.0</td>
      <td>11767</td>
      <td>Pave</td>
      <td>5</td>
      <td>6</td>
      <td>1946</td>
      <td>1995</td>
      <td>1200</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>430</th>
      <td>21.0</td>
      <td>1680</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1971</td>
      <td>1971</td>
      <td>987</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>513</th>
      <td>71.0</td>
      <td>9187</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1983</td>
      <td>1983</td>
      <td>1080</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>40.0</td>
      <td>3951</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2009</td>
      <td>2009</td>
      <td>1224</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>330</th>
      <td>NaN</td>
      <td>10624</td>
      <td>Pave</td>
      <td>5</td>
      <td>4</td>
      <td>1964</td>
      <td>1964</td>
      <td>1728</td>
      <td>2</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>NaN</td>
      <td>11</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>63.0</td>
      <td>13072</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>1141</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>121</th>
      <td>50.0</td>
      <td>6060</td>
      <td>Pave</td>
      <td>4</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1123</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
      <td>6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
<p>512 rows × 15 columns</p>
</div>



Looks good, still 512 records. So, NaN fireplace quality is not actually information that is missing from our dataset, it is a genuine category which means "fireplace quality is not applicable". This interpretation aligns with what we see in `data/data_description.txt`:

```
...
FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
...
```

Eventually we will still need to perform some preprocessing to prepare the `FireplaceQu` column for modeling (because models require numeric inputs), but we don't need to worry about filling in missing values.

### Lot Frontage

Now let's look at `LotFrontage` — it's possible that NaN is also a genuine category here, and it's possible that it's just missing data instead. Let's apply some domain understanding to understand whether it's possible that lot frontage can be N/A just like fireplace quality can be N/A.

Lot frontage is defined as the "Linear feet of street connected to property", i.e. how much of the property runs directly along a road. The amount of frontage required for a property depends on its zoning. Let's look at the zoning of all records with NaN for `LotFrontage`:


```python
# Run this cell without changes
df[df["LotFrontage"].isna()]["MSZoning"].value_counts()
```


```python
# __SOLUTION__
df[df["LotFrontage"].isna()]["MSZoning"].value_counts()
```




    RL    229
    RM     19
    FV      8
    RH      3
    Name: MSZoning, dtype: int64



So, we have RL (residential low density), RM (residential medium density), FV (floating village residential), and RH (residential high density). Looking at the building codes from the City of Ames, it appears that all of these zones require at least 24 feet of frontage.

Nevertheless, we can't assume that all properties have frontage just because the zoning regulations require it. Maybe these properties predate the regulations, or they received some kind of variance permitting them to get around the requirement. **It's still not as clear here as it was with the fireplaces whether this is a genuine "not applicable" kind of NaN or a "missing information" kind of NaN.**

In a case like this, we can take a double approach:

1. Make a new column in the dataset that simply represents whether `LotFrontage` was originally NaN
2. Fill in the NaN values of `LotFrontage` with median frontage in preparation for modeling

### Missing Indicator for `LotFrontage`

First, we import `sklearn.impute.MissingIndicator` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html)). The goal of using a `MissingIndicator` is creating a new column to represent which values were NaN (or some other "missing" value) in the original dataset, in case NaN ends up being a meaningful indicator rather than a random missing bit of data.

A `MissingIndicator` is a scikit-learn transformer, meaning that we will use the standard steps for any scikit-learn transformer:

1. Identify data to be transformed (typically not every column is passed to every transformer)
2. Instantiate the transformer object
3. Fit the transformer object (on training data only)
4. Transform data using the transformer object
5. Add the transformed data to the other data that was not transformed


```python
# Replace None with appropriate code
from sklearn.impute import MissingIndicator

# (1) Identify data to be transformed
# We only want missing indicators for LotFrontage
frontage_train = X_train[["LotFrontage"]]

# (2) Instantiate the transformer object
missing_indicator = MissingIndicator()

# (3) Fit the transformer object on frontage_train
None

# (4) Transform frontage_train and assign the result
# to frontage_missing_train
frontage_missing_train = None

# Visually inspect frontage_missing_train
frontage_missing_train
```


```python
# __SOLUTION__
from sklearn.impute import MissingIndicator

# (1) Identify data to be transformed
# We only want missing indicators for LotFrontage
frontage_train = X_train[["LotFrontage"]]

# (2) Instantiate the transformer object
missing_indicator = MissingIndicator()

# (3) Fit the transformer object on frontage_train
missing_indicator.fit(frontage_train)

# (4) Transform frontage_train and assign the result
# to frontage_missing_train
frontage_missing_train = missing_indicator.transform(frontage_train)

# Visually inspect frontage_missing_train
frontage_missing_train
```




    array([[False],
           [False],
           [False],
           ...,
           [False],
           [False],
           [False]])



The result of transforming `frontage_train` should be an array of arrays, each containing `True` or `False`. Make sure the `assert`s pass before moving on to the next step.


```python
# Run this cell without changes
import numpy as np

# frontage_missing_train should be a NumPy array
assert type(frontage_missing_train) == np.ndarray

# We should have the same number of rows as the full X_train
assert frontage_missing_train.shape[0] == X_train.shape[0]

# But we should only have 1 column
assert frontage_missing_train.shape[1] == 1
```


```python
# __SOLUTION__
import numpy as np

# frontage_missing_train should be a NumPy array
assert type(frontage_missing_train) == np.ndarray

# We should have the same number of rows as the full X_train
assert frontage_missing_train.shape[0] == X_train.shape[0]

# But we should only have 1 column
assert frontage_missing_train.shape[1] == 1
```

Now let's add this new information as a new column of `X_train`:


```python
# Run this cell without changes

# (5) add the transformed data to the other data
X_train["LotFrontage_Missing"] = frontage_missing_train
X_train
```


```python
# __SOLUTION__

# (5) add the transformed data to the other data
X_train["LotFrontage_Missing"] = frontage_missing_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
      <td>False</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>626</th>
      <td>NaN</td>
      <td>12342</td>
      <td>Pave</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
      <td>True</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>Pave</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>Pave</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>




```python
# Run this cell without changes

# Now we should have 1 extra column compared to
# our original subset
assert X_train.shape[1] == len(relevant_columns) + 1
```


```python
# __SOLUTION__

# Now we should have 1 extra column compared to
# our original subset
assert X_train.shape[1] == len(relevant_columns) + 1
```

### Imputing Missing Values for LotFrontage

Now that we have noted where missing values were originally present, let's use a `SimpleImputer` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)) to fill in those NaNs in the `LotFrontage` column.

The process is very similar to the `MissingIndicator` process, except that we want to replace the original `LotFrontage` column with the transformed version instead of just adding a new column on.

In the cell below, create and use a `SimpleImputer` with `strategy="median"` to transform the value of `frontage_train` (declared above).


```python
# Replace None with appropriate code

from sklearn.impute import SimpleImputer

# (1) frontage_train was created previously, so we don't
# need to extract the relevant data again

# (2) Instantiate a SimpleImputer with strategy="median"
imputer = None

# (3) Fit the imputer on frontage_train
None

# (4) Transform frontage_train using the imputer and
# assign the result to frontage_imputed_train
frontage_imputed_train = None

# Visually inspect frontage_imputed_train
frontage_imputed_train
```


```python
# __SOLUTION__

from sklearn.impute import SimpleImputer

# (1) frontage_train was created previously, so we don't
# need to extract the relevant data again

# (2) Instantiate a SimpleImputer with strategy="median"
imputer = SimpleImputer(strategy="median")

# (3) Fit the imputer on frontage_train
imputer.fit(frontage_train)

# (4) Transform frontage_train using the imputer and
# assign the result to frontage_imputed_train
frontage_imputed_train = imputer.transform(frontage_train)

# Visually inspect frontage_imputed_train
frontage_imputed_train
```




    array([[43.],
           [78.],
           [60.],
           ...,
           [60.],
           [55.],
           [53.]])



Now we can replace the original value of `LotFrontage` in `X_train` with the new value:


```python
# Run this cell without changes

# (5) Replace value of LotFrontage
X_train["LotFrontage"] = frontage_imputed_train

# Visually inspect X_train
X_train
```


```python
# __SOLUTION__

# (5) Replace value of LotFrontage
X_train["LotFrontage"] = frontage_imputed_train

# Visually inspect X_train
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>2008</td>
      <td>False</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>Fa</td>
      <td>1</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>Pave</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>TA</td>
      <td>8</td>
      <td>2007</td>
      <td>True</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>Pave</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>Pave</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>Pave</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>Pave</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>Pave</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>2007</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>Pave</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>2009</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>



Now the shape of `X_train` should still be the same as before:


```python
# Run this cell without changes
assert X_train.shape == (1095, 16)
```


```python
# __SOLUTION__
assert X_train.shape == (1095, 16)
```

And now our only NaN values should be in `FireplaceQu`, which are NaN values but not missing values:


```python
# Run this cell without changes
X_train.isna().sum()
```


```python
# __SOLUTION__
X_train.isna().sum()
```




    LotFrontage              0
    LotArea                  0
    Street                   0
    OverallQual              0
    OverallCond              0
    YearBuilt                0
    YearRemodAdd             0
    GrLivArea                0
    FullBath                 0
    BedroomAbvGr             0
    TotRmsAbvGrd             0
    Fireplaces               0
    FireplaceQu            512
    MoSold                   0
    YrSold                   0
    LotFrontage_Missing      0
    dtype: int64



Great! Now we have completed Step 2.


```python

```
