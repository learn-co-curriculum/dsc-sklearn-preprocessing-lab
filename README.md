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

#### 6. Preprocess Test Data

Apply Steps 1-5 to the test data in order to perform a final model evaluation.

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

So, let's replace those NaNs with the string "N/A" to indicate that this is a real category, not missing data:


```python
# Run this cell without changes
X_train["FireplaceQu"] = X_train["FireplaceQu"].fillna("N/A")
X_train["FireplaceQu"].value_counts()
```


```python
# __SOLUTION__
X_train["FireplaceQu"] = X_train["FireplaceQu"].fillna("N/A")
X_train["FireplaceQu"].value_counts()
```




    N/A    512
    Gd     286
    TA     236
    Fa      26
    Ex      19
    Po      16
    Name: FireplaceQu, dtype: int64



Eventually we will still need to perform some preprocessing to prepare the `FireplaceQu` column for modeling (because models require numeric inputs rather than inputs of type `object`), but we don't need to worry about filling in missing values.

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
      <td>N/A</td>
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
      <td>N/A</td>
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
      <td>N/A</td>
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
      <td>N/A</td>
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
      <td>N/A</td>
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
      <td>N/A</td>
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

And now our `X_train` no longer contains any NaN values:


```python
# Run this cell without changes
X_train.isna().sum()
```


```python
# __SOLUTION__
X_train.isna().sum()
```




    LotFrontage            0
    LotArea                0
    Street                 0
    OverallQual            0
    OverallCond            0
    YearBuilt              0
    YearRemodAdd           0
    GrLivArea              0
    FullBath               0
    BedroomAbvGr           0
    TotRmsAbvGrd           0
    Fireplaces             0
    FireplaceQu            0
    MoSold                 0
    YrSold                 0
    LotFrontage_Missing    0
    dtype: int64



Great! Now we have completed Step 2.

## 3. Convert Categorical Features into Numbers

Despite dropping irrelevant columns and filling in those NaN values, if we feed the current `X_train` into our scikit-learn `ElasticNet` model, it will crash:


```python
# Run this cell without changes
model.fit(X_train, y_train)
```


```python
# __SOLUTION__
model.fit(X_train, y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-5a4f775ea40e> in <module>
          1 # __SOLUTION__
    ----> 2 model.fit(X_train, y_train)
    

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


    ValueError: could not convert string to float: 'Pave'


Now the first column to cause a problem is `Street`, which is documented like this:

```
...
Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
...
```

Let's look at the full `X_train`:


```python
# Run this cell without changes
X_train.info()
```


```python
# __SOLUTION__
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   object 
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  FireplaceQu          1095 non-null   object 
     13  MoSold               1095 non-null   int64  
     14  YrSold               1095 non-null   int64  
     15  LotFrontage_Missing  1095 non-null   bool   
    dtypes: bool(1), float64(1), int64(12), object(2)
    memory usage: 137.9+ KB


So, our model is crashing because some of the columns are non-numeric.

Anything that is already `float64` or `int64` will work with our model, but these features need to be converted:

* `Street` (currently type `object`)
* `FireplaceQu` (currently type `object`)
* `LotFrontage_Missing` (currently type `bool`)

There are two main approaches to converting these values, depending on whether there are 2 values (meaning the categorical variable can be converted into a single binary number) or more than 2 values (meaning we need to create extra columns to represent all categories). (If there is only 1 value, this is not a useful feature for the purposes of predictive analysis.)

In the cell below, we inspect the value counts of the specified features:


```python
# Run this cell without changes

print(X_train["Street"].value_counts())
print()
print(X_train["FireplaceQu"].value_counts())
print()
print(X_train["LotFrontage_Missing"].value_counts())
```


```python
# __SOLUTION__

print(X_train["Street"].value_counts())
print()
print(X_train["FireplaceQu"].value_counts())
print()
print(X_train["LotFrontage_Missing"].value_counts())
```

    Pave    1091
    Grvl       4
    Name: Street, dtype: int64
    
    N/A    512
    Gd     286
    TA     236
    Fa      26
    Ex      19
    Po      16
    Name: FireplaceQu, dtype: int64
    
    False    895
    True     200
    Name: LotFrontage_Missing, dtype: int64


So, it looks like `Street` and `LotFrontage_Missing` have only 2 categories and can be converted into binary in place, whereas `FireplaceQu` has 6 categories and will need to be expanded into multiple columns.

### Binary Categories

For binary categories, we will use `LabelBinarizer` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)) to convert the categories of `Street` and `LotFrontage_Missing` into binary values (0s and 1s).

Just like in Step 2 when we used the `MissingIndicator` and `SimpleImputer`, we will follow these steps:

1. Identify data to be transformed
2. Instantiate the transformer object
3. Fit the transformer object (on training data only)
4. Transform data using the transformer object
5. Add the transformed data to the other data that was not transformed

Let's start with transforming `Street`:


```python
# Replace None with appropriate code

# (0) import LabelBinarizer from sklearn.preprocessing
None

# (1) Create a variable street_train that is the
# relevant column from X_train
street_train = None

# (2) Instantiate a LabelBinarizer
binarizer_street = None

# (3) Fit the binarizer on street_train
None

# Inspect the classes of the fitted binarizer
binarizer_street.classes_
```


```python
# __SOLUTION__

# (0) import LabelBinarizer from sklearn.preprocessing
from sklearn.preprocessing import LabelBinarizer

# (1) Create a variable street_train that is the
# relevant column from X_train
street_train = X_train["Street"]

# (2) Instantiate a LabelBinarizer
binarizer_street = LabelBinarizer()

# (3) Fit the binarizer on street_train
binarizer_street.fit(street_train)

# Inspect the classes of the fitted binarizer
binarizer_street.classes_
```




    array(['Grvl', 'Pave'], dtype='<U4')



The `.classes_` attribute of `LabelBinarizer` is only present once the `.fit` method has been called. (The trailing `_` indicates this convention.)

What this tells us is that when `binarizer_street` is used to transform the street data into 1s and 0s, `0` will mean `'Grvl'` (gravel) in the original data, and `1` will mean `'Pave'` (paved) in the original data.

The eventual scikit-learn model only cares about the 1s and 0s, but this information can be useful for us to understand what our code is doing and help us debug when things go wrong.

Now let's transform `street_train` with the fitted binarizer:


```python
# Replace None with appropriate code

# (4) Transform street_train using the binarizer and
# assign the result to street_binarized_train
street_binarized_train = None

# Visually inspect street_binarized_train
street_binarized_train
```


```python
# __SOLUTION__

# (4) Transform street_train using the binarizer and
# assign the result to street_binarized_train
street_binarized_train = binarizer_street.transform(street_train)

# Visually inspect street_binarized_train
street_binarized_train
```




    array([[1],
           [1],
           [1],
           ...,
           [1],
           [1],
           [1]])



All of the values we see appear to be `1` right now, but that makes sense since there were only 4 properties with gravel (`0`) streets in `X_train`.

Now let's replace the original `Street` column with the binarized version:


```python
# Replace None with appropriate code

# (5) Replace value of Street
X_train["Street"] = None

# Visually inspect X_train
X_train
```


```python
# __SOLUTION__

# (5) Replace value of Street
X_train["Street"] = street_binarized_train

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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>N/A</td>
      <td>10</td>
      <td>2009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1</td>
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
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>N/A</td>
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
      <td>1</td>
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
      <td>1</td>
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
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2006</td>
      <td>False</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1</td>
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
      <td>1</td>
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
X_train.info()
```


```python
# __SOLUTION__
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   int64  
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  FireplaceQu          1095 non-null   object 
     13  MoSold               1095 non-null   int64  
     14  YrSold               1095 non-null   int64  
     15  LotFrontage_Missing  1095 non-null   bool   
    dtypes: bool(1), float64(1), int64(13), object(1)
    memory usage: 137.9+ KB


Perfect! Now `Street` should by type `int64` instead of `object`.

Now, repeat the same process with `LotFrontage_Missing`:


```python
# Replace None with appropriate code

# (1) We already have a variable frontage_missing_train
# from earlier, no additional step needed

# (2) Instantiate a LabelBinarizer for missing frontage
binarizer_frontage_missing = None

# (3) Fit the binarizer on frontage_missing_train
None

# Inspect the classes of the fitted binarizer
binarizer_frontage_missing.classes_
```


```python
# __SOLUTION__

# (1) We already have a variable frontage_missing_train
# from earlier, no additional step needed

# (2) Instantiate a LabelBinarizer for missing frontage
binarizer_frontage_missing = LabelBinarizer()

# (3) Fit the binarizer on frontage_missing_train
binarizer_frontage_missing.fit(frontage_missing_train)

# Inspect the classes of the fitted binarizer
binarizer_frontage_missing.classes_
```




    array([False,  True])




```python
# Replace None with appropriate code

# (4) Transform frontage_missing_train using the binarizer and
# assign the result to frontage_missing_binarized_train
frontage_missing_binarized_train = None

# Visually inspect frontage_missing_binarized_train
frontage_missing_binarized_train
```


```python
# __SOLUTION__

# (4) Transform frontage_missing_train using the binarizer and
# assign the result to frontage_missing_binarized_train
frontage_missing_binarized_train = binarizer_frontage_missing.transform(frontage_missing_train)

# Visually inspect frontage_missing_binarized_train
frontage_missing_binarized_train
```




    array([[0],
           [0],
           [0],
           ...,
           [0],
           [0],
           [0]])




```python
# Replace None with appropriate code

# (5) Replace value of LotFrontage_Missing
X_train["LotFrontage_Missing"] = None

# Visually inspect X_train
X_train
```


```python
# __SOLUTION__

# (5) Replace value of LotFrontage_Missing
X_train["LotFrontage_Missing"] = frontage_missing_binarized_train

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
      <td>1</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>1</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>N/A</td>
      <td>10</td>
      <td>2009</td>
      <td>0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1</td>
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
      <td>1</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2007</td>
      <td>0</td>
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
      <td>1</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>1</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>N/A</td>
      <td>4</td>
      <td>2006</td>
      <td>0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>1</td>
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
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>




```python
# Run this cell without changes
X_train.info()
```


```python
# __SOLUTION__
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   int64  
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  FireplaceQu          1095 non-null   object 
     13  MoSold               1095 non-null   int64  
     14  YrSold               1095 non-null   int64  
     15  LotFrontage_Missing  1095 non-null   int64  
    dtypes: float64(1), int64(14), object(1)
    memory usage: 145.4+ KB


Great, now we only have 1 column remaining that isn't type `float64` or `int64`!

#### Note on Preprocessing Boolean Values
For binary values like `LotFrontage_Missing`, you might see a few different approaches to preprocessing. Python treats `True` and `1` as equal:


```python
# Run this cell without changes
print(True == 1)
print(False == 0)
```


```python
# __SOLUTION__
print(True == 1)
print(False == 0)
```

    True
    True


This means that if your model is purely using Python, you actually might just be able to leave columns as type `bool` without any issues. You will likely see examples that do this. However if your model relies on C or Java "under the hood", this might cause problems.

There is also a technique using `pandas` rather than scikit-learn for this particular conversion of boolean values to integers:


```python
# Run this cell without changes
df_example = pd.DataFrame(frontage_missing_train, columns=["LotFrontage_Missing"])
df_example
```


```python
# __SOLUTION__
df_example = pd.DataFrame(frontage_missing_train, columns=["LotFrontage_Missing"])
df_example
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
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>False</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 1 columns</p>
</div>




```python
# Run this cell without changes
df_example["LotFrontage_Missing"] = df_example["LotFrontage_Missing"].astype(int)
df_example
```


```python
# __SOLUTION__
df_example["LotFrontage_Missing"] = df_example["LotFrontage_Missing"].astype(int)
df_example
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
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 1 columns</p>
</div>



This code is casting every value in the `LotFrontage_Missing` column to an integer, achieving the same result as the `LabelBinarizer` example with less code.

The downside of using this approach is that it doesn't fit into a scikit-learn pipeline as neatly because it is using `pandas` to do the transformation instead of scikit-learn.

In the future, you will need to make your own determination of which strategy to use!

### Multiple Categories

Unlike `Street` and `LotFrontage_Missing`, `FireplaceQu` has more than two categories. Therefore the process for encoding it numerically is a bit more complicated, because we will need to create multiple "dummy" columns that are each representing one category.

To do this, we can use a `OneHotEncoder` from `sklearn.preprocessing` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)).

The first several steps are very similar to all of the other transformers we've used so far, although the process of combining the data with the original data differs.

In the cells below, complete steps `(0)`-`(4)` of preprocessing the `FireplaceQu` column using a `OneHotEncoder`:


```python
# Replace None with appropriate code

# (0) import OneHotEncoder from sklearn.preprocessing
None

# (1) Create a variable fireplace_qu_train
# extracted from X_train
# (double brackets due to shape expected by OHE)
fireplace_qu_train = X_train[["FireplaceQu"]]

# (2) Instantiate a OneHotEncoder with categories="auto",
# sparse=False, and handle_unknown="ignore"
ohe = None

# (3) Fit the encoder on fireplace_qu_train
None

# Inspect the categories of the fitted encoder
ohe.categories_
```


```python
# __SOLUTION__

# (0) import OneHotEncoder from sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder

# (1) Create a variable fireplace_qu_train
# extracted from X_train
# (double brackets due to shape expected by OHE)
fireplace_qu_train = X_train[["FireplaceQu"]]

# (2) Instantiate a OneHotEncoder with categories="auto",
# sparse=False, and handle_unknown="ignore"
ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")

# (3) Fit the encoder on fireplace_qu_train
ohe.fit(fireplace_qu_train)

# Inspect the categories of the fitted encoder
ohe.categories_
```




    [array(['Ex', 'Fa', 'Gd', 'N/A', 'Po', 'TA'], dtype=object)]




```python
# Replace None with appropriate code

# (4) Transform fireplace_qu_train using the encoder and
# assign the result to fireplace_qu_encoded_train
fireplace_qu_encoded_train = None

# Visually inspect fireplace_qu_encoded_train
fireplace_qu_encoded_train
```


```python
# __SOLUTION__

# (4) Transform fireplace_qu_train using the encoder and
# assign the result to fireplace_qu_encoded_train
fireplace_qu_encoded_train = ohe.transform(fireplace_qu_train)

# Visually inspect fireplace_qu_encoded_train
fireplace_qu_encoded_train
```




    array([[0., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           ...,
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1.]])



Notice that this time, unlike with `MissingIndicator`, `SimpleImputer`, or `LabelBinarizer`, we have created multiple columns of data out of a single column. The code below converts this unlabeled NumPy array into a readable pandas dataframe in preparation for merging it back with the rest of `X_train`:


```python
# Run this cell without changes

# (5a) Make the transformed data into a dataframe
fireplace_qu_encoded_train = pd.DataFrame(
    # Pass in NumPy array
    fireplace_qu_encoded_train,
    # Set the column names to the categories found by OHE
    columns=ohe.categories_[0],
    # Set the index to match X_train's index
    index=X_train.index
)

# Visually inspect new dataframe
fireplace_qu_encoded_train
```


```python
# __SOLUTION__

# (5a) Make the transformed data into a dataframe
fireplace_qu_encoded_train = pd.DataFrame(
    # Pass in NumPy array
    fireplace_qu_encoded_train,
    # Set the column names to the categories found by OHE
    columns=ohe.categories_[0],
    # Set the index to match X_train's index
    index=X_train.index
)

# Visually inspect new dataframe
fireplace_qu_encoded_train
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
      <th>Ex</th>
      <th>Fa</th>
      <th>Gd</th>
      <th>N/A</th>
      <th>Po</th>
      <th>TA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 6 columns</p>
</div>



A couple notes on the code above:

* The main goal of converting this into a dataframe (rather than converting `X_train` into a NumPy array, which would also allow them to be combined) is **readability** — to help you and others understand what your code is doing, and to help you debug. Eventually when you write this code as a pipeline, it will be NumPy arrays "under the hood".
* We are using just the **raw categories** from `FireplaceQu` as our new dataframe columns, but you'll also see examples where a lambda function or list comprehension is used to create column names indicating the original column name, e.g. `FireplaceQu_Ex`, `FireplaceQu_Fa` rather than just `Ex`, `Fa`. This is a design decision based on readability — the scikit-learn model will work the same either way.
* It is very important that **the index of the new dataframe matches the index of the main `X_train` dataframe**. Because we used `train_test_split`, the index of `X_train` is shuffled, so it goes `1023`, `810`, `1384` etc. instead of `0`, `1`, `2`, etc. If you don't specify an index for the new dataframe, it will assign the first record to the index `0` rather than `1023`. If you are ever merging encoded data like this and a bunch of NaNs start appearing, make sure that the indexes are lined up correctly! You also may see examples where the index of `X_train` has been reset, rather than specifying the index of the new dataframe — either way works.

Next, we want to drop the original `FireplaceQu` column containing the categorical data:

(For previous transformations we didn't need to drop anything because we were replacing 1 column with 1 new column in place, but one-hot encoding works differently.)


```python
# Run this cell without changes

# (5b) Drop original FireplaceQu column
X_train.drop("FireplaceQu", axis=1, inplace=True)

# Visually inspect X_train
X_train
```


```python
# __SOLUTION__

# (5b) Drop original FireplaceQu column
X_train.drop("FireplaceQu", axis=1, inplace=True)

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
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2006</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
      <td>0</td>
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
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>2007</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>12</td>
      <td>2009</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>2009</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



Finally, we want to concatenate the new dataframe together with the original `X_train`:


```python
# Run this cell without changes

# (5c) Concatenate the new dataframe with current X_train
X_train = pd.concat([X_train, fireplace_qu_encoded_train], axis=1)

# Visually inspect X_train
X_train
```


```python
# __SOLUTION__

# (5c) Concatenate the new dataframe with current X_train
X_train = pd.concat([X_train, fireplace_qu_encoded_train], axis=1)

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
      <th>...</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>Ex</th>
      <th>Fa</th>
      <th>Gd</th>
      <th>N/A</th>
      <th>Po</th>
      <th>TA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182</td>
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>1504</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1974</td>
      <td>1999</td>
      <td>1309</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2006</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1939</td>
      <td>1950</td>
      <td>1258</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1960</td>
      <td>1978</td>
      <td>1422</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1442</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>1095</th>
      <td>78.0</td>
      <td>9317</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>1314</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>2007</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1928</td>
      <td>1950</td>
      <td>1981</td>
      <td>2</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>12</td>
      <td>2009</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172</td>
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>1955</td>
      <td>1990</td>
      <td>864</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642</td>
      <td>1</td>
      <td>7</td>
      <td>8</td>
      <td>1918</td>
      <td>1998</td>
      <td>1426</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684</td>
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>1555</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>6</td>
      <td>2009</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 21 columns</p>
</div>




```python
# Run this cell without changes
X_train.info()
```


```python
# __SOLUTION__
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 21 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   LotFrontage          1095 non-null   float64
     1   LotArea              1095 non-null   int64  
     2   Street               1095 non-null   int64  
     3   OverallQual          1095 non-null   int64  
     4   OverallCond          1095 non-null   int64  
     5   YearBuilt            1095 non-null   int64  
     6   YearRemodAdd         1095 non-null   int64  
     7   GrLivArea            1095 non-null   int64  
     8   FullBath             1095 non-null   int64  
     9   BedroomAbvGr         1095 non-null   int64  
     10  TotRmsAbvGrd         1095 non-null   int64  
     11  Fireplaces           1095 non-null   int64  
     12  MoSold               1095 non-null   int64  
     13  YrSold               1095 non-null   int64  
     14  LotFrontage_Missing  1095 non-null   int64  
     15  Ex                   1095 non-null   float64
     16  Fa                   1095 non-null   float64
     17  Gd                   1095 non-null   float64
     18  N/A                  1095 non-null   float64
     19  Po                   1095 non-null   float64
     20  TA                   1095 non-null   float64
    dtypes: float64(7), int64(14)
    memory usage: 188.2 KB


Ok, everything is numeric now! We have completed the minimum necessary preprocessing to use these features in a scikit-learn model!


```python
# Run this cell without changes
model.fit(X_train, y_train)
```


```python
# __SOLUTION__
model.fit(X_train, y_train)
```




    ElasticNet(random_state=1)



Great, no error this time.

Let's use cross validation to take a look at the model's performance:


```python
# Run this cell without changes
from sklearn.model_selection import cross_val_score

cross_val_score(model, X_train, y_train, cv=3)
```


```python
# __SOLUTION__
from sklearn.model_selection import cross_val_score

cross_val_score(model, X_train, y_train, cv=3)
```




    array([0.73895092, 0.66213118, 0.8124859 ])



Not terrible, we are explaining between 66% and 81% of the variance in the target with our current feature set.

## 4. Add Interaction Terms

Now that we have completed the minimum preprocessing to run a model without errors, let's try to improve the model performance.

Linear models (including `ElasticNet`) are limited in the information they can learn because they are assuming a linear relationship between features and target. Often our real-world features aren't related to the target this way:


```python
# Run this cell without changes
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X_train["LotArea"], y_train, alpha=0.2)
ax.set_xlabel("Lot Area")
ax.set_ylabel("Sale Price")
ax.set_title("Lot Area vs. Sale Price for Ames Housing Data");
```


```python
# __SOLUTION__
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X_train["LotArea"], y_train, alpha=0.2)
ax.set_xlabel("Lot Area")
ax.set_ylabel("Sale Price")
ax.set_title("Lot Area vs. Sale Price for Ames Housing Data");
```


![png](index_files/index_133_0.png)


Sometimes we can improve the linearity by introducing an *interaction term*. In this case, multiplying the lot area by the overall quality:


```python
# Run this cell without changes

fig, ax = plt.subplots()
ax.scatter(X_train["LotArea"]*X_train["OverallQual"], y_train, alpha=0.2)
ax.set_xlabel("Lot Area x Overall Quality")
ax.set_ylabel("Sale Price")
ax.set_title("(Lot Area x Overall Quality) vs. Sale Price for Ames Housing Data");
```


```python
# __SOLUTION__

fig, ax = plt.subplots()
ax.scatter(X_train["LotArea"]*X_train["OverallQual"], y_train, alpha=0.2)
ax.set_xlabel("Lot Area x Overall Quality")
ax.set_ylabel("Sale Price")
ax.set_title("(Lot Area x Overall Quality) vs. Sale Price for Ames Housing Data");
```


![png](index_files/index_136_0.png)


While we could manually add individual interaction terms, there is a preprocessor from scikit-learn called `PolynomialFeatures` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)) that will generate all combinations of interaction terms (as well as polynomial terms, e.g. `Lot Area` squared) for a set of columns.

Specifically, let's generate interaction terms for `LotFrontage`, `LotArea`, `OverallQual`, `YearBuilt`, and `GrLivArea`.

To use `PolynomialFeatures`, we'll follow the same steps as `OneHotEncoder` (creating a new dataframe before merging), because it is another transformer that creates additional columns.


```python
# Replace None with appropriate code

# (0) import PolynomialFeatures from sklearn.preprocessing
None

# (1) Create a variable poly_columns
# extracted from X_train
poly_column_names = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "GrLivArea"
]
poly_columns_train = X_train[poly_column_names]

# (2) Instantiate a PolynomialFeatures transformer poly
# with interaction_only=True and include_bias=False
poly = None

# (3) Fit the transformer on poly_columns_train
None

# Inspect the features created by the fitted transformer
poly.get_feature_names(poly_column_names)
```


```python
# __SOLUTION__

# (0) import PolynomialFeatures from sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures

# (1) Create a variable poly_columns
# extracted from X_train
poly_column_names = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "GrLivArea"
]
poly_columns_train = X_train[poly_column_names]

# (2) Instantiate a PolynomialFeatures transformer
# with interaction_only=True and include_bias=False
poly = PolynomialFeatures(interaction_only=True, include_bias=False)

# (3) Fit the transformer on poly_columns
poly.fit(poly_columns_train)

# Inspect the features created by the fitted transformer
poly.get_feature_names(poly_column_names)
```




    ['LotFrontage',
     'LotArea',
     'OverallQual',
     'YearBuilt',
     'GrLivArea',
     'LotFrontage LotArea',
     'LotFrontage OverallQual',
     'LotFrontage YearBuilt',
     'LotFrontage GrLivArea',
     'LotArea OverallQual',
     'LotArea YearBuilt',
     'LotArea GrLivArea',
     'OverallQual YearBuilt',
     'OverallQual GrLivArea',
     'YearBuilt GrLivArea']




```python
# Replace None with appropriate code

# (4) Transform poly_columns using the transformer and
# assign the result poly_columns_expanded_train
poly_columns_expanded_train = None

# Visually inspect poly_columns_expanded_train
poly_columns_expanded_train
```


```python
# __SOLUTION__

# (4) Transform poly_columns using the transformer and
# assign the result poly_columns_expanded_train
poly_columns_expanded_train = poly.transform(poly_columns_train)

# Visually inspect poly_columns_expanded_train
poly_columns_expanded_train
```




    array([[4.300000e+01, 3.182000e+03, 7.000000e+00, ..., 1.403500e+04,
            1.052800e+04, 3.015520e+06],
           [7.800000e+01, 1.014000e+04, 6.000000e+00, ..., 1.184400e+04,
            7.854000e+03, 2.583966e+06],
           [6.000000e+01, 9.060000e+03, 6.000000e+00, ..., 1.163400e+04,
            7.548000e+03, 2.439262e+06],
           ...,
           [6.000000e+01, 8.172000e+03, 5.000000e+00, ..., 9.775000e+03,
            4.320000e+03, 1.689120e+06],
           [5.500000e+01, 7.642000e+03, 7.000000e+00, ..., 1.342600e+04,
            9.982000e+03, 2.735068e+06],
           [5.300000e+01, 3.684000e+03, 7.000000e+00, ..., 1.404900e+04,
            1.088500e+04, 3.120885e+06]])




```python
# Replace None with appropriate code

# (5a) Make the transformed data into a dataframe
poly_columns_expanded_train = pd.DataFrame(
    # Pass in NumPy array created in previous step
    None,
    # Set the column names to the features created by poly
    columns=poly.get_feature_names(poly_column_names),
    # Set the index to match X_train's index
    index=None
)

# Visually inspect new dataframe
poly_columns_expanded_train
```


```python
# __SOLUTION__

# (5a) Make the transformed data into a dataframe
poly_columns_expanded_train = pd.DataFrame(
    # Pass in NumPy array created in previous step
    poly_columns_expanded_train,
    # Set the column names to the features created by poly
    columns=poly.get_feature_names(poly_column_names),
    # Set the index to match X_train's index
    index=X_train.index
)

# Visually inspect new dataframe
poly_columns_expanded_train
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
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>GrLivArea</th>
      <th>LotFrontage LotArea</th>
      <th>LotFrontage OverallQual</th>
      <th>LotFrontage YearBuilt</th>
      <th>LotFrontage GrLivArea</th>
      <th>LotArea OverallQual</th>
      <th>LotArea YearBuilt</th>
      <th>LotArea GrLivArea</th>
      <th>OverallQual YearBuilt</th>
      <th>OverallQual GrLivArea</th>
      <th>YearBuilt GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>43.0</td>
      <td>3182.0</td>
      <td>7.0</td>
      <td>2005.0</td>
      <td>1504.0</td>
      <td>136826.0</td>
      <td>301.0</td>
      <td>86215.0</td>
      <td>64672.0</td>
      <td>22274.0</td>
      <td>6379910.0</td>
      <td>4785728.0</td>
      <td>14035.0</td>
      <td>10528.0</td>
      <td>3015520.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>78.0</td>
      <td>10140.0</td>
      <td>6.0</td>
      <td>1974.0</td>
      <td>1309.0</td>
      <td>790920.0</td>
      <td>468.0</td>
      <td>153972.0</td>
      <td>102102.0</td>
      <td>60840.0</td>
      <td>20016360.0</td>
      <td>13273260.0</td>
      <td>11844.0</td>
      <td>7854.0</td>
      <td>2583966.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>60.0</td>
      <td>9060.0</td>
      <td>6.0</td>
      <td>1939.0</td>
      <td>1258.0</td>
      <td>543600.0</td>
      <td>360.0</td>
      <td>116340.0</td>
      <td>75480.0</td>
      <td>54360.0</td>
      <td>17567340.0</td>
      <td>11397480.0</td>
      <td>11634.0</td>
      <td>7548.0</td>
      <td>2439262.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>70.0</td>
      <td>12342.0</td>
      <td>5.0</td>
      <td>1960.0</td>
      <td>1422.0</td>
      <td>863940.0</td>
      <td>350.0</td>
      <td>137200.0</td>
      <td>99540.0</td>
      <td>61710.0</td>
      <td>24190320.0</td>
      <td>17550324.0</td>
      <td>9800.0</td>
      <td>7110.0</td>
      <td>2787120.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>75.0</td>
      <td>9750.0</td>
      <td>6.0</td>
      <td>1958.0</td>
      <td>1442.0</td>
      <td>731250.0</td>
      <td>450.0</td>
      <td>146850.0</td>
      <td>108150.0</td>
      <td>58500.0</td>
      <td>19090500.0</td>
      <td>14059500.0</td>
      <td>11748.0</td>
      <td>8652.0</td>
      <td>2823436.0</td>
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
      <td>9317.0</td>
      <td>6.0</td>
      <td>2006.0</td>
      <td>1314.0</td>
      <td>726726.0</td>
      <td>468.0</td>
      <td>156468.0</td>
      <td>102492.0</td>
      <td>55902.0</td>
      <td>18689902.0</td>
      <td>12242538.0</td>
      <td>12036.0</td>
      <td>7884.0</td>
      <td>2635884.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>65.0</td>
      <td>7804.0</td>
      <td>4.0</td>
      <td>1928.0</td>
      <td>1981.0</td>
      <td>507260.0</td>
      <td>260.0</td>
      <td>125320.0</td>
      <td>128765.0</td>
      <td>31216.0</td>
      <td>15046112.0</td>
      <td>15459724.0</td>
      <td>7712.0</td>
      <td>7924.0</td>
      <td>3819368.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>60.0</td>
      <td>8172.0</td>
      <td>5.0</td>
      <td>1955.0</td>
      <td>864.0</td>
      <td>490320.0</td>
      <td>300.0</td>
      <td>117300.0</td>
      <td>51840.0</td>
      <td>40860.0</td>
      <td>15976260.0</td>
      <td>7060608.0</td>
      <td>9775.0</td>
      <td>4320.0</td>
      <td>1689120.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>55.0</td>
      <td>7642.0</td>
      <td>7.0</td>
      <td>1918.0</td>
      <td>1426.0</td>
      <td>420310.0</td>
      <td>385.0</td>
      <td>105490.0</td>
      <td>78430.0</td>
      <td>53494.0</td>
      <td>14657356.0</td>
      <td>10897492.0</td>
      <td>13426.0</td>
      <td>9982.0</td>
      <td>2735068.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>53.0</td>
      <td>3684.0</td>
      <td>7.0</td>
      <td>2007.0</td>
      <td>1555.0</td>
      <td>195252.0</td>
      <td>371.0</td>
      <td>106371.0</td>
      <td>82415.0</td>
      <td>25788.0</td>
      <td>7393788.0</td>
      <td>5728620.0</td>
      <td>14049.0</td>
      <td>10885.0</td>
      <td>3120885.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>




```python
# Run this cell without changes

# (5b) Drop original columns
X_train.drop(poly_column_names, axis=1, inplace=True)

# (5c) Concatenate the new dataframe with current X_train
X_train = pd.concat([X_train, poly_columns_expanded_train], axis=1)

# Visually inspect X_train
X_train
```


```python
# __SOLUTION__

# (5b) Drop original columns
X_train.drop(poly_column_names, axis=1, inplace=True)

# (5c) Concatenate the new dataframe with current X_train
X_train = pd.concat([X_train, poly_columns_expanded_train], axis=1)

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
      <th>Street</th>
      <th>OverallCond</th>
      <th>YearRemodAdd</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>...</th>
      <th>LotFrontage LotArea</th>
      <th>LotFrontage OverallQual</th>
      <th>LotFrontage YearBuilt</th>
      <th>LotFrontage GrLivArea</th>
      <th>LotArea OverallQual</th>
      <th>LotArea YearBuilt</th>
      <th>LotArea GrLivArea</th>
      <th>OverallQual YearBuilt</th>
      <th>OverallQual GrLivArea</th>
      <th>YearBuilt GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>1</td>
      <td>5</td>
      <td>2006</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>0</td>
      <td>...</td>
      <td>136826.0</td>
      <td>301.0</td>
      <td>86215.0</td>
      <td>64672.0</td>
      <td>22274.0</td>
      <td>6379910.0</td>
      <td>4785728.0</td>
      <td>14035.0</td>
      <td>10528.0</td>
      <td>3015520.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>1</td>
      <td>6</td>
      <td>1999</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2006</td>
      <td>0</td>
      <td>...</td>
      <td>790920.0</td>
      <td>468.0</td>
      <td>153972.0</td>
      <td>102102.0</td>
      <td>60840.0</td>
      <td>20016360.0</td>
      <td>13273260.0</td>
      <td>11844.0</td>
      <td>7854.0</td>
      <td>2583966.0</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>1</td>
      <td>5</td>
      <td>1950</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>0</td>
      <td>...</td>
      <td>543600.0</td>
      <td>360.0</td>
      <td>116340.0</td>
      <td>75480.0</td>
      <td>54360.0</td>
      <td>17567340.0</td>
      <td>11397480.0</td>
      <td>11634.0</td>
      <td>7548.0</td>
      <td>2439262.0</td>
    </tr>
    <tr>
      <th>626</th>
      <td>1</td>
      <td>5</td>
      <td>1978</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1</td>
      <td>...</td>
      <td>863940.0</td>
      <td>350.0</td>
      <td>137200.0</td>
      <td>99540.0</td>
      <td>61710.0</td>
      <td>24190320.0</td>
      <td>17550324.0</td>
      <td>9800.0</td>
      <td>7110.0</td>
      <td>2787120.0</td>
    </tr>
    <tr>
      <th>813</th>
      <td>1</td>
      <td>6</td>
      <td>1958</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>2007</td>
      <td>0</td>
      <td>...</td>
      <td>731250.0</td>
      <td>450.0</td>
      <td>146850.0</td>
      <td>108150.0</td>
      <td>58500.0</td>
      <td>19090500.0</td>
      <td>14059500.0</td>
      <td>11748.0</td>
      <td>8652.0</td>
      <td>2823436.0</td>
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
      <th>1095</th>
      <td>1</td>
      <td>5</td>
      <td>2006</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>2007</td>
      <td>0</td>
      <td>...</td>
      <td>726726.0</td>
      <td>468.0</td>
      <td>156468.0</td>
      <td>102492.0</td>
      <td>55902.0</td>
      <td>18689902.0</td>
      <td>12242538.0</td>
      <td>12036.0</td>
      <td>7884.0</td>
      <td>2635884.0</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>1</td>
      <td>3</td>
      <td>1950</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>12</td>
      <td>2009</td>
      <td>0</td>
      <td>...</td>
      <td>507260.0</td>
      <td>260.0</td>
      <td>125320.0</td>
      <td>128765.0</td>
      <td>31216.0</td>
      <td>15046112.0</td>
      <td>15459724.0</td>
      <td>7712.0</td>
      <td>7924.0</td>
      <td>3819368.0</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>1</td>
      <td>7</td>
      <td>1990</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>0</td>
      <td>...</td>
      <td>490320.0</td>
      <td>300.0</td>
      <td>117300.0</td>
      <td>51840.0</td>
      <td>40860.0</td>
      <td>15976260.0</td>
      <td>7060608.0</td>
      <td>9775.0</td>
      <td>4320.0</td>
      <td>1689120.0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>1</td>
      <td>8</td>
      <td>1998</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>0</td>
      <td>...</td>
      <td>420310.0</td>
      <td>385.0</td>
      <td>105490.0</td>
      <td>78430.0</td>
      <td>53494.0</td>
      <td>14657356.0</td>
      <td>10897492.0</td>
      <td>13426.0</td>
      <td>9982.0</td>
      <td>2735068.0</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>1</td>
      <td>5</td>
      <td>2007</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>2009</td>
      <td>0</td>
      <td>...</td>
      <td>195252.0</td>
      <td>371.0</td>
      <td>106371.0</td>
      <td>82415.0</td>
      <td>25788.0</td>
      <td>7393788.0</td>
      <td>5728620.0</td>
      <td>14049.0</td>
      <td>10885.0</td>
      <td>3120885.0</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 31 columns</p>
</div>




```python
# Run this cell without changes
X_train.info()
```


```python
# __SOLUTION__
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1095 entries, 1023 to 1126
    Data columns (total 31 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   Street                   1095 non-null   int64  
     1   OverallCond              1095 non-null   int64  
     2   YearRemodAdd             1095 non-null   int64  
     3   FullBath                 1095 non-null   int64  
     4   BedroomAbvGr             1095 non-null   int64  
     5   TotRmsAbvGrd             1095 non-null   int64  
     6   Fireplaces               1095 non-null   int64  
     7   MoSold                   1095 non-null   int64  
     8   YrSold                   1095 non-null   int64  
     9   LotFrontage_Missing      1095 non-null   int64  
     10  Ex                       1095 non-null   float64
     11  Fa                       1095 non-null   float64
     12  Gd                       1095 non-null   float64
     13  N/A                      1095 non-null   float64
     14  Po                       1095 non-null   float64
     15  TA                       1095 non-null   float64
     16  LotFrontage              1095 non-null   float64
     17  LotArea                  1095 non-null   float64
     18  OverallQual              1095 non-null   float64
     19  YearBuilt                1095 non-null   float64
     20  GrLivArea                1095 non-null   float64
     21  LotFrontage LotArea      1095 non-null   float64
     22  LotFrontage OverallQual  1095 non-null   float64
     23  LotFrontage YearBuilt    1095 non-null   float64
     24  LotFrontage GrLivArea    1095 non-null   float64
     25  LotArea OverallQual      1095 non-null   float64
     26  LotArea YearBuilt        1095 non-null   float64
     27  LotArea GrLivArea        1095 non-null   float64
     28  OverallQual YearBuilt    1095 non-null   float64
     29  OverallQual GrLivArea    1095 non-null   float64
     30  YearBuilt GrLivArea      1095 non-null   float64
    dtypes: float64(21), int64(10)
    memory usage: 273.8 KB


Great, now we have 31 features instead of 21 features! Let's see how the model performs now:


```python
# Run this cell without changes
cross_val_score(model, X_train, y_train, cv=3)
```


```python
# __SOLUTION__
cross_val_score(model, X_train, y_train, cv=3)
```

    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:529: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 298739023179.6966, tolerance: 422550782.65263027
      model = cd_fast.enet_coordinate_descent(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:529: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 357401730994.73425, tolerance: 434287718.1245009
      model = cd_fast.enet_coordinate_descent(
    //anaconda3/envs/learn-env/lib/python3.8/site-packages/sklearn/linear_model/_coordinate_descent.py:529: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 402015942401.8634, tolerance: 471766118.83873975
      model = cd_fast.enet_coordinate_descent(





    array([0.75336526, 0.79206309, 0.75227628])



Hmm, got some metrics, so it didn't totally crash, but what is that warning message?

A `ConvergenceWarning` means that the **gradient descent** algorithm within the `ElasticNet` model failed to find a minimum based on the specified parameters. While the warning message suggests modifyig the parameters (number of iterations), your first thought when you see a model fail to converge should be **do I need to scale the data**?

Scaling data is especially important when there are substantial differences in the units of different features. Let's take a look at the values in our current `X_train`:


```python
# Run this cell without changes
X_train.describe()
```


```python
# __SOLUTION__
X_train.describe()
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
      <th>Street</th>
      <th>OverallCond</th>
      <th>YearRemodAdd</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>...</th>
      <th>LotFrontage LotArea</th>
      <th>LotFrontage OverallQual</th>
      <th>LotFrontage YearBuilt</th>
      <th>LotFrontage GrLivArea</th>
      <th>LotArea OverallQual</th>
      <th>LotArea YearBuilt</th>
      <th>LotArea GrLivArea</th>
      <th>OverallQual YearBuilt</th>
      <th>OverallQual GrLivArea</th>
      <th>YearBuilt GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1095.000000</td>
      <td>1095.00000</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>...</td>
      <td>1.095000e+03</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1095.000000</td>
      <td>1095.000000</td>
      <td>1.095000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.996347</td>
      <td>5.56895</td>
      <td>1984.854795</td>
      <td>1.578995</td>
      <td>2.896804</td>
      <td>6.564384</td>
      <td>0.619178</td>
      <td>6.361644</td>
      <td>2007.818265</td>
      <td>0.182648</td>
      <td>...</td>
      <td>8.308276e+05</td>
      <td>438.181735</td>
      <td>138690.935160</td>
      <td>1.119997e+05</td>
      <td>6.737676e+04</td>
      <td>2.118694e+07</td>
      <td>1.800923e+07</td>
      <td>12105.757078</td>
      <td>9805.720548</td>
      <td>3.020889e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.060357</td>
      <td>1.10448</td>
      <td>20.732472</td>
      <td>0.544976</td>
      <td>0.806361</td>
      <td>1.625103</td>
      <td>0.644338</td>
      <td>2.680894</td>
      <td>1.325752</td>
      <td>0.386555</td>
      <td>...</td>
      <td>1.350715e+06</td>
      <td>207.779337</td>
      <td>45385.148925</td>
      <td>7.993630e+04</td>
      <td>7.490171e+04</td>
      <td>2.176917e+07</td>
      <td>2.596924e+07</td>
      <td>2806.986068</td>
      <td>5199.725503</td>
      <td>1.047711e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.101700e+04</td>
      <td>50.000000</td>
      <td>41370.000000</td>
      <td>1.323000e+04</td>
      <td>5.000000e+03</td>
      <td>2.574000e+06</td>
      <td>9.305100e+05</td>
      <td>1922.000000</td>
      <td>334.000000</td>
      <td>6.499640e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>5.00000</td>
      <td>1966.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>4.756920e+05</td>
      <td>312.000000</td>
      <td>116400.000000</td>
      <td>7.078750e+04</td>
      <td>4.034000e+04</td>
      <td>1.494741e+07</td>
      <td>8.749200e+06</td>
      <td>9810.000000</td>
      <td>5945.000000</td>
      <td>2.259095e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>5.00000</td>
      <td>1994.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>6.653500e+05</td>
      <td>420.000000</td>
      <td>137760.000000</td>
      <td>1.001600e+05</td>
      <td>5.670500e+04</td>
      <td>1.877812e+07</td>
      <td>1.368276e+07</td>
      <td>11832.000000</td>
      <td>8892.000000</td>
      <td>2.918504e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>6.00000</td>
      <td>2004.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>9.050150e+05</td>
      <td>510.000000</td>
      <td>156000.000000</td>
      <td>1.360195e+05</td>
      <td>7.761550e+04</td>
      <td>2.302560e+07</td>
      <td>2.041942e+07</td>
      <td>14021.000000</td>
      <td>12372.500000</td>
      <td>3.543578e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>9.00000</td>
      <td>2010.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>3.228675e+07</td>
      <td>3130.000000</td>
      <td>628504.000000</td>
      <td>1.765946e+06</td>
      <td>1.506715e+06</td>
      <td>4.229564e+08</td>
      <td>4.382388e+08</td>
      <td>20090.000000</td>
      <td>56420.000000</td>
      <td>1.132914e+07</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>



Looks like we have mean values ranging from about $0.6$ (for fireplaces) to $2.1 x 10^7$ (21 million, for `Lot Area x YearBuilt`). With the regularization applied by `ElasticNet`, the coefficients are being penalized very disproportionately!

In the next step, we'll apply scaling to address this.

## 5. Scale Data

This is the final scikit-learn preprocessing task of the lab! The `StandardScaler` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)) standarizes features by removing the mean and scaling to unit variance. This will help our data to meet the assumptions of the L1 and L2 regularizers in our `ElasticNet` model.

Unlike previous preprocessing steps, we are going to apply the `StandardScaler` to the entire `X_train`, not just a single column or a subset of columns.


```python
# Replace None with appropriate code

# (0) import StandardScaler from sklearn.preprocessing
None

# (1) We don't actually have to select anything since 
# we're using the full X_train

# (2) Instantiate a StandardScaler
scaler = None

# (3) Fit the scaler on X_train
None

# (4) Transform X_train using the scaler and
# assign the result to X_train_scaled
X_train_scaled = None

# Visually inspect X_train_scaled
X_train_scaled
```


```python
# __SOLUTION__

# (0) import StandardScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

# (1) We don't actually have to select anything since 
# we're using the full X_train

# (2) Instantiate a StandardScaler
scaler = StandardScaler()

# (3) Fit the scaler on X_train
scaler.fit(X_train)

# (4) Transform X_train using the scaler and
# assign the result to X_train_scaled
X_train_scaled = scaler.transform(X_train)

# Visually inspect X_train_scaled
X_train_scaled
```




    array([[ 0.06055048, -0.51536449,  1.02037363, ...,  0.68761455,
             0.1389707 , -0.00512705],
           [ 0.06055048,  0.39045271,  0.68258474, ..., -0.09329461,
            -0.3755222 , -0.41721713],
           [ 0.06055048, -0.51536449, -1.68193746, ..., -0.16814214,
            -0.43439835, -0.55539469],
           ...,
           [ 0.06055048,  1.29626991,  0.24828475, ..., -0.83072093,
            -1.05548402, -1.27170382],
           [ 0.06055048,  2.20208711,  0.63432919, ...,  0.47055673,
             0.03391718, -0.27293012],
           [ 0.06055048, -0.51536449,  1.06862918, ...,  0.69260438,
             0.20765954,  0.09548578]])




```python
# Run this cell without changes

# (5) Make the transformed data back into a dataframe
X_train = pd.DataFrame(
    # Pass in NumPy array
    X_train_scaled,
    # Set the column names to the original names
    columns=X_train.columns,
    # Set the index to match X_train's original index
    index=X_train.index
)

# Visually inspect new dataframe
X_train
```


```python
# __SOLUTION__

# (5) Make the transformed data back into a dataframe
X_train = pd.DataFrame(
    # Pass in NumPy array
    X_train_scaled,
    # Set the column names to the original names
    columns=X_train.columns,
    # Set the index to match X_train's original index
    index=X_train.index
)

# Visually inspect new dataframe
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
      <th>Street</th>
      <th>OverallCond</th>
      <th>YearRemodAdd</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>...</th>
      <th>LotFrontage LotArea</th>
      <th>LotFrontage OverallQual</th>
      <th>LotFrontage YearBuilt</th>
      <th>LotFrontage GrLivArea</th>
      <th>LotArea OverallQual</th>
      <th>LotArea YearBuilt</th>
      <th>LotArea GrLivArea</th>
      <th>OverallQual YearBuilt</th>
      <th>OverallQual GrLivArea</th>
      <th>YearBuilt GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>1.020374</td>
      <td>0.772872</td>
      <td>-1.112669</td>
      <td>0.268177</td>
      <td>0.591298</td>
      <td>-0.508139</td>
      <td>0.137143</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.514038</td>
      <td>-0.660530</td>
      <td>-1.156764</td>
      <td>-0.592338</td>
      <td>-0.602434</td>
      <td>-0.680494</td>
      <td>-0.509431</td>
      <td>0.687615</td>
      <td>0.138971</td>
      <td>-0.005127</td>
    </tr>
    <tr>
      <th>810</th>
      <td>0.06055</td>
      <td>0.390453</td>
      <td>0.682585</td>
      <td>-1.062909</td>
      <td>0.128036</td>
      <td>-0.963076</td>
      <td>0.591298</td>
      <td>-2.000860</td>
      <td>-1.372124</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.029559</td>
      <td>0.143575</td>
      <td>0.336851</td>
      <td>-0.123876</td>
      <td>-0.087311</td>
      <td>-0.053797</td>
      <td>-0.182452</td>
      <td>-0.093295</td>
      <td>-0.375522</td>
      <td>-0.417217</td>
    </tr>
    <tr>
      <th>1384</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>-1.681937</td>
      <td>-1.062909</td>
      <td>-1.112669</td>
      <td>-0.347450</td>
      <td>-0.961392</td>
      <td>1.357763</td>
      <td>0.891777</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.212746</td>
      <td>-0.376445</td>
      <td>-0.492697</td>
      <td>-0.457069</td>
      <td>-0.173864</td>
      <td>-0.166348</td>
      <td>-0.254716</td>
      <td>-0.168142</td>
      <td>-0.434398</td>
      <td>-0.555395</td>
    </tr>
    <tr>
      <th>626</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>-0.330782</td>
      <td>-1.062909</td>
      <td>0.128036</td>
      <td>-0.347450</td>
      <td>0.591298</td>
      <td>0.611402</td>
      <td>-0.617490</td>
      <td>2.115420</td>
      <td>...</td>
      <td>0.024526</td>
      <td>-0.424595</td>
      <td>-0.032866</td>
      <td>-0.155942</td>
      <td>-0.075691</td>
      <td>0.138028</td>
      <td>-0.017679</td>
      <td>-0.821811</td>
      <td>-0.518672</td>
      <td>-0.223226</td>
    </tr>
    <tr>
      <th>813</th>
      <td>0.06055</td>
      <td>0.390453</td>
      <td>-1.295893</td>
      <td>-1.062909</td>
      <td>1.368742</td>
      <td>0.268177</td>
      <td>-0.961392</td>
      <td>-0.881319</td>
      <td>-0.617490</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.073756</td>
      <td>0.056905</td>
      <td>0.179856</td>
      <td>-0.048182</td>
      <td>-0.118566</td>
      <td>-0.096347</td>
      <td>-0.152162</td>
      <td>-0.127511</td>
      <td>-0.221982</td>
      <td>-0.188548</td>
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
      <th>1095</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>1.020374</td>
      <td>0.772872</td>
      <td>0.128036</td>
      <td>-0.347450</td>
      <td>0.591298</td>
      <td>-1.254499</td>
      <td>-0.617490</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.077107</td>
      <td>0.143575</td>
      <td>0.391872</td>
      <td>-0.118995</td>
      <td>-0.153268</td>
      <td>-0.114758</td>
      <td>-0.222160</td>
      <td>-0.024863</td>
      <td>-0.369750</td>
      <td>-0.367641</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>0.06055</td>
      <td>-2.326999</td>
      <td>-1.681937</td>
      <td>0.772872</td>
      <td>1.368742</td>
      <td>0.268177</td>
      <td>2.143989</td>
      <td>2.104124</td>
      <td>0.891777</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.239662</td>
      <td>-0.857945</td>
      <td>-0.294745</td>
      <td>0.209829</td>
      <td>-0.482997</td>
      <td>-0.282217</td>
      <td>-0.098219</td>
      <td>-1.566009</td>
      <td>-0.362054</td>
      <td>0.762466</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>0.06055</td>
      <td>1.296270</td>
      <td>0.248285</td>
      <td>-1.062909</td>
      <td>-1.112669</td>
      <td>-0.963076</td>
      <td>-0.961392</td>
      <td>-0.881319</td>
      <td>-1.372124</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.252209</td>
      <td>-0.665345</td>
      <td>-0.471536</td>
      <td>-0.752939</td>
      <td>-0.354183</td>
      <td>-0.239470</td>
      <td>-0.421792</td>
      <td>-0.830721</td>
      <td>-1.055484</td>
      <td>-1.271704</td>
    </tr>
    <tr>
      <th>860</th>
      <td>0.06055</td>
      <td>2.202087</td>
      <td>0.634329</td>
      <td>-1.062909</td>
      <td>0.128036</td>
      <td>0.268177</td>
      <td>0.591298</td>
      <td>-0.134958</td>
      <td>-0.617490</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.304065</td>
      <td>-0.256070</td>
      <td>-0.731872</td>
      <td>-0.420148</td>
      <td>-0.185431</td>
      <td>-0.300083</td>
      <td>-0.273978</td>
      <td>0.470557</td>
      <td>0.033917</td>
      <td>-0.272930</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>1.068629</td>
      <td>0.772872</td>
      <td>-1.112669</td>
      <td>0.268177</td>
      <td>0.591298</td>
      <td>-0.134958</td>
      <td>0.891777</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.470762</td>
      <td>-0.323480</td>
      <td>-0.712451</td>
      <td>-0.370273</td>
      <td>-0.555498</td>
      <td>-0.633899</td>
      <td>-0.473107</td>
      <td>0.692604</td>
      <td>0.207660</td>
      <td>0.095486</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 31 columns</p>
</div>




```python
# Run this cell without changes
X_train.describe()
```


```python
# __SOLUTION__
X_train.describe()
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
      <th>Street</th>
      <th>OverallCond</th>
      <th>YearRemodAdd</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>...</th>
      <th>LotFrontage LotArea</th>
      <th>LotFrontage OverallQual</th>
      <th>LotFrontage YearBuilt</th>
      <th>LotFrontage GrLivArea</th>
      <th>LotArea OverallQual</th>
      <th>LotArea YearBuilt</th>
      <th>LotArea GrLivArea</th>
      <th>OverallQual YearBuilt</th>
      <th>OverallQual GrLivArea</th>
      <th>YearBuilt GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>...</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
      <td>1.095000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.617603e-16</td>
      <td>-2.984928e-16</td>
      <td>-3.202309e-15</td>
      <td>-6.488975e-18</td>
      <td>2.530700e-16</td>
      <td>-6.813423e-17</td>
      <td>-1.946692e-17</td>
      <td>7.624545e-17</td>
      <td>6.328535e-14</td>
      <td>9.084565e-17</td>
      <td>...</td>
      <td>2.757814e-17</td>
      <td>-1.492464e-16</td>
      <td>2.984928e-16</td>
      <td>-5.191180e-17</td>
      <td>-4.380058e-17</td>
      <td>8.111218e-18</td>
      <td>6.326750e-17</td>
      <td>2.271141e-17</td>
      <td>-8.435667e-17</td>
      <td>-1.460019e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>...</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
      <td>1.000457e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.651514e+01</td>
      <td>-4.138633e+00</td>
      <td>-1.681937e+00</td>
      <td>-2.898690e+00</td>
      <td>-3.594081e+00</td>
      <td>-2.809956e+00</td>
      <td>-9.613917e-01</td>
      <td>-2.000860e+00</td>
      <td>-1.372124e+00</td>
      <td>-4.727195e-01</td>
      <td>...</td>
      <td>-5.924092e-01</td>
      <td>-1.869094e+00</td>
      <td>-2.145314e+00</td>
      <td>-1.236170e+00</td>
      <td>-8.331620e-01</td>
      <td>-8.554045e-01</td>
      <td>-6.579525e-01</td>
      <td>-3.629662e+00</td>
      <td>-1.822413e+00</td>
      <td>-2.263992e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.055048e-02</td>
      <td>-5.153645e-01</td>
      <td>-9.098486e-01</td>
      <td>-1.062909e+00</td>
      <td>-1.112669e+00</td>
      <td>-9.630763e-01</td>
      <td>-9.613917e-01</td>
      <td>-5.081387e-01</td>
      <td>-6.174901e-01</td>
      <td>-4.727195e-01</td>
      <td>...</td>
      <td>-2.630443e-01</td>
      <td>-6.075647e-01</td>
      <td>-4.913748e-01</td>
      <td>-5.157986e-01</td>
      <td>-3.611281e-01</td>
      <td>-2.867533e-01</td>
      <td>-3.567399e-01</td>
      <td>-8.182463e-01</td>
      <td>-7.428247e-01</td>
      <td>-7.274358e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.055048e-02</td>
      <td>-5.153645e-01</td>
      <td>4.413070e-01</td>
      <td>7.728723e-01</td>
      <td>1.280363e-01</td>
      <td>-3.474496e-01</td>
      <td>5.912984e-01</td>
      <td>-1.349584e-01</td>
      <td>1.371434e-01</td>
      <td>-4.727195e-01</td>
      <td>...</td>
      <td>-1.225671e-01</td>
      <td>-8.754500e-02</td>
      <td>-2.052126e-02</td>
      <td>-1.481819e-01</td>
      <td>-1.425420e-01</td>
      <td>-1.107031e-01</td>
      <td>-1.666760e-01</td>
      <td>-9.757162e-02</td>
      <td>-1.758051e-01</td>
      <td>-9.776744e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.055048e-02</td>
      <td>3.904527e-01</td>
      <td>9.238625e-01</td>
      <td>7.728723e-01</td>
      <td>1.280363e-01</td>
      <td>2.681771e-01</td>
      <td>5.912984e-01</td>
      <td>6.114023e-01</td>
      <td>8.917769e-01</td>
      <td>-4.727195e-01</td>
      <td>...</td>
      <td>5.494964e-02</td>
      <td>3.458047e-01</td>
      <td>3.815560e-01</td>
      <td>3.006240e-01</td>
      <td>1.367581e-01</td>
      <td>8.450031e-02</td>
      <td>9.285201e-02</td>
      <td>6.826247e-01</td>
      <td>4.938631e-01</td>
      <td>4.991145e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.055048e-02</td>
      <td>3.107904e+00</td>
      <td>1.213396e+00</td>
      <td>2.608653e+00</td>
      <td>6.331565e+00</td>
      <td>4.577564e+00</td>
      <td>3.696679e+00</td>
      <td>2.104124e+00</td>
      <td>1.646410e+00</td>
      <td>2.115420e+00</td>
      <td>...</td>
      <td>2.329899e+01</td>
      <td>1.296110e+01</td>
      <td>1.079730e+01</td>
      <td>2.070026e+01</td>
      <td>1.922514e+01</td>
      <td>1.846433e+01</td>
      <td>1.618922e+01</td>
      <td>2.845718e+00</td>
      <td>8.968854e+00</td>
      <td>7.933529e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>



Great, now the means of the values are all much more centered. Let's see how the model performs now:


```python
# Run this cell without changes
cross_val_score(model, X_train, y_train, cv=3)
```


```python
# __SOLUTION__
cross_val_score(model, X_train, y_train, cv=3)
```




    array([0.75126826, 0.67341002, 0.80080353])



Well, that was only a minor improvement over our first model. Seems like the interaction terms didn't provide that much information after all! There is plenty more feature engineering you could do if this were a real project, but we'll stop there.

### Quick Scaling FAQs:

1. **Do you only need to scale if you're using `PolynomialFeatures`?** No, we should have scaled regardless. `PolynomialFeatures` just exaggerated the difference in the units and caused the model to produce a warning, but it's a best practice to scale whenever your model has any distance-based metric. (In this case, the regularization within `ElasticNet` is distance-based.)
2. **Do you really need to scale one-hot encoded features, if they are already just 0 or 1?** Professional opinions vary on this. Binary values already violate the assumptions of some models, so you might want to investigate empirically with your particular data and model whether you get better performance by scaling the one-hot encoded features or leaving them as just 0 and 1.

## 6. Preprocess Test Data

> Apply Steps 1-5 to the test data in order to perform a final model evaluation.

This part is done for you, and it should work automatically, assuming you didn't change the names of any of the transformer objects. Note that we are intentionally **not instantiating or fitting the transformers** here, because you always want to fit transformers on the training data only.

*Step 1: Drop Irrelevant Columns*


```python
# Run this cell without changes
X_test = X_test.loc[:, relevant_columns]
```


```python
# __SOLUTION__
X_test = X_test.loc[:, relevant_columns]
```

*Step 2: Handle Missing Values*


```python
# Run this cell without changes

# Replace FireplaceQu NaNs with "N/A"s
X_test["FireplaceQu"] = X_test["FireplaceQu"].fillna("N/A")

# Add missing indicator for lot frontage
frontage_test = X_test[["LotFrontage"]]
frontage_missing_test = missing_indicator.transform(frontage_test)
X_test["LotFrontage_Missing"] = frontage_missing_test

# Impute missing lot frontage values
frontage_imputed_test = imputer.transform(frontage_test)
X_test["LotFrontage"] = frontage_imputed_test

# Check that there are no more missing values
X_test.isna().sum()
```


```python
# __SOLUTION__

# Replace FireplaceQu NaNs with "N/A"s
X_test["FireplaceQu"] = X_test["FireplaceQu"].fillna("N/A")

# Add missing indicator for lot frontage
frontage_test = X_test[["LotFrontage"]]
frontage_missing_test = missing_indicator.transform(frontage_test)
X_test["LotFrontage_Missing"] = frontage_missing_test

# Impute missing lot frontage values
frontage_imputed_test = imputer.transform(frontage_test)
X_test["LotFrontage"] = frontage_imputed_test

# Check that there are no more missing values
X_test.isna().sum()
```




    LotFrontage            0
    LotArea                0
    Street                 0
    OverallQual            0
    OverallCond            0
    YearBuilt              0
    YearRemodAdd           0
    GrLivArea              0
    FullBath               0
    BedroomAbvGr           0
    TotRmsAbvGrd           0
    Fireplaces             0
    FireplaceQu            0
    MoSold                 0
    YrSold                 0
    LotFrontage_Missing    0
    dtype: int64



*Step 3: Convert Categorical Features into Numbers*


```python
# Run this cell without changes

# Binarize street type
street_test = X_test["Street"]
street_binarized_test = binarizer_street.transform(street_test)
X_test["Street"] = street_binarized_test

# Binarize frontage missing
frontage_missing_test = X_test["LotFrontage_Missing"]
frontage_missing_binarized_test = binarizer_frontage_missing.transform(frontage_missing_test)
X_test["LotFrontage_Missing"] = frontage_missing_binarized_test

# One-hot encode fireplace quality
fireplace_qu_test = X_test[["FireplaceQu"]]
fireplace_qu_encoded_test = ohe.transform(fireplace_qu_test)
fireplace_qu_encoded_test = pd.DataFrame(
    fireplace_qu_encoded_test,
    columns=ohe.categories_[0],
    index=X_test.index
)
X_test.drop("FireplaceQu", axis=1, inplace=True)
X_test = pd.concat([X_test, fireplace_qu_encoded_test], axis=1)

# Visually inspect X_test
X_test
```


```python
# __SOLUTION__

# Binarize street type
street_test = X_test["Street"]
street_binarized_test = binarizer_street.transform(street_test)
X_test["Street"] = street_binarized_test

# Binarize frontage missing
frontage_missing_test = X_test["LotFrontage_Missing"]
frontage_missing_binarized_test = binarizer_frontage_missing.transform(frontage_missing_test)
X_test["LotFrontage_Missing"] = frontage_missing_binarized_test

# One-hot encode fireplace quality
fireplace_qu_test = X_test[["FireplaceQu"]]
fireplace_qu_encoded_test = ohe.transform(fireplace_qu_test)
fireplace_qu_encoded_test = pd.DataFrame(
    fireplace_qu_encoded_test,
    columns=ohe.categories_[0],
    index=X_test.index
)
X_test.drop("FireplaceQu", axis=1, inplace=True)
X_test = pd.concat([X_test, fireplace_qu_encoded_test], axis=1)

# Visually inspect X_test
X_test
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
      <th>...</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>Ex</th>
      <th>Fa</th>
      <th>Gd</th>
      <th>N/A</th>
      <th>Po</th>
      <th>TA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>70.0</td>
      <td>8414</td>
      <td>1</td>
      <td>6</td>
      <td>8</td>
      <td>1963</td>
      <td>2003</td>
      <td>1068</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>98.0</td>
      <td>12256</td>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>1994</td>
      <td>1995</td>
      <td>2622</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>4</td>
      <td>2010</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>56.0</td>
      <td>8960</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>1927</td>
      <td>1950</td>
      <td>1028</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>2010</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>522</th>
      <td>50.0</td>
      <td>5000</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>1947</td>
      <td>1950</td>
      <td>1664</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>10</td>
      <td>2006</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>89.0</td>
      <td>12898</td>
      <td>1</td>
      <td>9</td>
      <td>5</td>
      <td>2007</td>
      <td>2008</td>
      <td>1620</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>9</td>
      <td>2009</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>988</th>
      <td>70.0</td>
      <td>12046</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>2030</td>
      <td>2</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>243</th>
      <td>75.0</td>
      <td>10762</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1980</td>
      <td>1980</td>
      <td>1217</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>4</td>
      <td>2009</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1342</th>
      <td>70.0</td>
      <td>9375</td>
      <td>1</td>
      <td>8</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>2169</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>70.0</td>
      <td>29959</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>1994</td>
      <td>1994</td>
      <td>1850</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2009</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>71.0</td>
      <td>9204</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1963</td>
      <td>1963</td>
      <td>1144</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 21 columns</p>
</div>



*Step 4: Add Interaction Terms*


```python
# Run this cell without changes

# (1) select relevant data
poly_columns_test = X_test[poly_column_names]

# (4) transform using fitted transformer
poly_columns_expanded_test = poly.transform(poly_columns_test) 

# (5) add back to original dataset
poly_columns_expanded_test = pd.DataFrame(
    poly_columns_expanded_test,
    columns=poly.get_feature_names(poly_column_names),
    index=X_test.index
)
X_test.drop(poly_column_names, axis=1, inplace=True)
X_test = pd.concat([X_test, poly_columns_expanded_test], axis=1)

# Visually inspect X_test
X_test
```


```python
# __SOLUTION__

# (1) select relevant data
poly_columns_test = X_test[poly_column_names]

# (4) transform using fitted transformer
poly_columns_expanded_test = poly.transform(poly_columns_test) 

# (5) add back to original dataset
poly_columns_expanded_test = pd.DataFrame(
    poly_columns_expanded_test,
    columns=poly.get_feature_names(poly_column_names),
    index=X_test.index
)
X_test.drop(poly_column_names, axis=1, inplace=True)
X_test = pd.concat([X_test, poly_columns_expanded_test], axis=1)

# Visually inspect X_test
X_test
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
      <th>Street</th>
      <th>OverallCond</th>
      <th>YearRemodAdd</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>...</th>
      <th>LotFrontage LotArea</th>
      <th>LotFrontage OverallQual</th>
      <th>LotFrontage YearBuilt</th>
      <th>LotFrontage GrLivArea</th>
      <th>LotArea OverallQual</th>
      <th>LotArea YearBuilt</th>
      <th>LotArea GrLivArea</th>
      <th>OverallQual YearBuilt</th>
      <th>OverallQual GrLivArea</th>
      <th>YearBuilt GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>1</td>
      <td>8</td>
      <td>2003</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>0</td>
      <td>...</td>
      <td>588980.0</td>
      <td>420.0</td>
      <td>137410.0</td>
      <td>74760.0</td>
      <td>50484.0</td>
      <td>16516682.0</td>
      <td>8986152.0</td>
      <td>11778.0</td>
      <td>6408.0</td>
      <td>2096484.0</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>1</td>
      <td>5</td>
      <td>1995</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>2010</td>
      <td>0</td>
      <td>...</td>
      <td>1201088.0</td>
      <td>784.0</td>
      <td>195412.0</td>
      <td>256956.0</td>
      <td>98048.0</td>
      <td>24438464.0</td>
      <td>32135232.0</td>
      <td>15952.0</td>
      <td>20976.0</td>
      <td>5228268.0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1</td>
      <td>6</td>
      <td>1950</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>2010</td>
      <td>0</td>
      <td>...</td>
      <td>501760.0</td>
      <td>280.0</td>
      <td>107912.0</td>
      <td>57568.0</td>
      <td>44800.0</td>
      <td>17265920.0</td>
      <td>9210880.0</td>
      <td>9635.0</td>
      <td>5140.0</td>
      <td>1980956.0</td>
    </tr>
    <tr>
      <th>522</th>
      <td>1</td>
      <td>7</td>
      <td>1950</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>2</td>
      <td>10</td>
      <td>2006</td>
      <td>0</td>
      <td>...</td>
      <td>250000.0</td>
      <td>300.0</td>
      <td>97350.0</td>
      <td>83200.0</td>
      <td>30000.0</td>
      <td>9735000.0</td>
      <td>8320000.0</td>
      <td>11682.0</td>
      <td>9984.0</td>
      <td>3239808.0</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>1</td>
      <td>5</td>
      <td>2008</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>9</td>
      <td>2009</td>
      <td>0</td>
      <td>...</td>
      <td>1147922.0</td>
      <td>801.0</td>
      <td>178623.0</td>
      <td>144180.0</td>
      <td>116082.0</td>
      <td>25886286.0</td>
      <td>20894760.0</td>
      <td>18063.0</td>
      <td>14580.0</td>
      <td>3251340.0</td>
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
      <th>988</th>
      <td>1</td>
      <td>6</td>
      <td>1976</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>6</td>
      <td>2007</td>
      <td>1</td>
      <td>...</td>
      <td>843220.0</td>
      <td>420.0</td>
      <td>138320.0</td>
      <td>142100.0</td>
      <td>72276.0</td>
      <td>23802896.0</td>
      <td>24453380.0</td>
      <td>11856.0</td>
      <td>12180.0</td>
      <td>4011280.0</td>
    </tr>
    <tr>
      <th>243</th>
      <td>1</td>
      <td>6</td>
      <td>1980</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>2009</td>
      <td>0</td>
      <td>...</td>
      <td>807150.0</td>
      <td>450.0</td>
      <td>148500.0</td>
      <td>91275.0</td>
      <td>64572.0</td>
      <td>21308760.0</td>
      <td>13097354.0</td>
      <td>11880.0</td>
      <td>7302.0</td>
      <td>2409660.0</td>
    </tr>
    <tr>
      <th>1342</th>
      <td>1</td>
      <td>5</td>
      <td>2002</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>2007</td>
      <td>1</td>
      <td>...</td>
      <td>656250.0</td>
      <td>560.0</td>
      <td>140140.0</td>
      <td>151830.0</td>
      <td>75000.0</td>
      <td>18768750.0</td>
      <td>20334375.0</td>
      <td>16016.0</td>
      <td>17352.0</td>
      <td>4342338.0</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>1</td>
      <td>6</td>
      <td>1994</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>2009</td>
      <td>1</td>
      <td>...</td>
      <td>2097130.0</td>
      <td>490.0</td>
      <td>139580.0</td>
      <td>129500.0</td>
      <td>209713.0</td>
      <td>59738246.0</td>
      <td>55424150.0</td>
      <td>13958.0</td>
      <td>12950.0</td>
      <td>3688900.0</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>1</td>
      <td>5</td>
      <td>1963</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
      <td>0</td>
      <td>...</td>
      <td>653484.0</td>
      <td>355.0</td>
      <td>139373.0</td>
      <td>81224.0</td>
      <td>46020.0</td>
      <td>18067452.0</td>
      <td>10529376.0</td>
      <td>9815.0</td>
      <td>5720.0</td>
      <td>2245672.0</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 31 columns</p>
</div>



*Step 5: Scale Data*


```python
# Run this cell without changes
X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)
X_test
```


```python
# __SOLUTION__
X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)
X_test
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
      <th>Street</th>
      <th>OverallCond</th>
      <th>YearRemodAdd</th>
      <th>FullBath</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>LotFrontage_Missing</th>
      <th>...</th>
      <th>LotFrontage LotArea</th>
      <th>LotFrontage OverallQual</th>
      <th>LotFrontage YearBuilt</th>
      <th>LotFrontage GrLivArea</th>
      <th>LotArea OverallQual</th>
      <th>LotArea YearBuilt</th>
      <th>LotArea GrLivArea</th>
      <th>OverallQual YearBuilt</th>
      <th>OverallQual GrLivArea</th>
      <th>YearBuilt GrLivArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>0.06055</td>
      <td>2.202087</td>
      <td>0.875607</td>
      <td>-1.062909</td>
      <td>0.128036</td>
      <td>-0.347450</td>
      <td>-0.961392</td>
      <td>-1.627680</td>
      <td>-1.372124</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.179133</td>
      <td>-0.087545</td>
      <td>-0.028237</td>
      <td>-0.466080</td>
      <td>-0.225635</td>
      <td>-0.214633</td>
      <td>-0.347611</td>
      <td>-0.116818</td>
      <td>-0.653741</td>
      <td>-0.882713</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>0.489563</td>
      <td>0.772872</td>
      <td>0.128036</td>
      <td>1.499430</td>
      <td>2.143989</td>
      <td>-0.881319</td>
      <td>1.646410</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>0.274247</td>
      <td>1.665114</td>
      <td>1.250343</td>
      <td>1.814226</td>
      <td>0.409674</td>
      <td>0.149432</td>
      <td>0.544200</td>
      <td>1.370866</td>
      <td>2.149226</td>
      <td>2.107822</td>
    </tr>
    <tr>
      <th>413</th>
      <td>0.06055</td>
      <td>0.390453</td>
      <td>-1.681937</td>
      <td>-1.062909</td>
      <td>-1.112669</td>
      <td>-0.963076</td>
      <td>0.591298</td>
      <td>-1.254499</td>
      <td>1.646410</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.243736</td>
      <td>-0.761645</td>
      <td>-0.678482</td>
      <td>-0.681250</td>
      <td>-0.301556</td>
      <td>-0.180200</td>
      <td>-0.338954</td>
      <td>-0.880619</td>
      <td>-0.897711</td>
      <td>-0.993030</td>
    </tr>
    <tr>
      <th>522</th>
      <td>0.06055</td>
      <td>1.296270</td>
      <td>-1.681937</td>
      <td>0.772872</td>
      <td>0.128036</td>
      <td>0.268177</td>
      <td>2.143989</td>
      <td>1.357763</td>
      <td>-1.372124</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.430211</td>
      <td>-0.665345</td>
      <td>-0.911307</td>
      <td>-0.360448</td>
      <td>-0.499239</td>
      <td>-0.526303</td>
      <td>-0.373275</td>
      <td>-0.151034</td>
      <td>0.034302</td>
      <td>0.209045</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>1.116885</td>
      <td>0.772872</td>
      <td>-1.112669</td>
      <td>-0.347450</td>
      <td>0.591298</td>
      <td>0.984583</td>
      <td>0.891777</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>0.234868</td>
      <td>1.746969</td>
      <td>0.880251</td>
      <td>0.402758</td>
      <td>0.650552</td>
      <td>0.215970</td>
      <td>0.111164</td>
      <td>2.123261</td>
      <td>0.918599</td>
      <td>0.220057</td>
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
      <th>988</th>
      <td>0.06055</td>
      <td>0.390453</td>
      <td>-0.427293</td>
      <td>0.772872</td>
      <td>1.368742</td>
      <td>0.883804</td>
      <td>0.591298</td>
      <td>-0.134958</td>
      <td>-0.617490</td>
      <td>2.115420</td>
      <td>...</td>
      <td>0.009179</td>
      <td>-0.087545</td>
      <td>-0.008177</td>
      <td>0.376726</td>
      <td>0.065439</td>
      <td>0.120223</td>
      <td>0.248259</td>
      <td>-0.089018</td>
      <td>0.456825</td>
      <td>0.945722</td>
    </tr>
    <tr>
      <th>243</th>
      <td>0.06055</td>
      <td>0.390453</td>
      <td>-0.234271</td>
      <td>-1.062909</td>
      <td>0.128036</td>
      <td>-0.347450</td>
      <td>0.591298</td>
      <td>-0.881319</td>
      <td>0.891777</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.017538</td>
      <td>0.056905</td>
      <td>0.216228</td>
      <td>-0.259384</td>
      <td>-0.037463</td>
      <td>0.005599</td>
      <td>-0.189229</td>
      <td>-0.080464</td>
      <td>-0.481730</td>
      <td>-0.583662</td>
    </tr>
    <tr>
      <th>1342</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>0.827351</td>
      <td>0.772872</td>
      <td>0.128036</td>
      <td>0.268177</td>
      <td>0.591298</td>
      <td>0.611402</td>
      <td>-0.617490</td>
      <td>2.115420</td>
      <td>...</td>
      <td>-0.129307</td>
      <td>0.586555</td>
      <td>0.031943</td>
      <td>0.498503</td>
      <td>0.101823</td>
      <td>-0.111134</td>
      <td>0.089575</td>
      <td>1.393676</td>
      <td>1.451947</td>
      <td>1.261849</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>0.06055</td>
      <td>0.390453</td>
      <td>0.441307</td>
      <td>0.772872</td>
      <td>0.128036</td>
      <td>0.268177</td>
      <td>0.591298</td>
      <td>-2.000860</td>
      <td>0.891777</td>
      <td>2.115420</td>
      <td>...</td>
      <td>0.937934</td>
      <td>0.249505</td>
      <td>0.019598</td>
      <td>0.219028</td>
      <td>1.901175</td>
      <td>1.771722</td>
      <td>1.441398</td>
      <td>0.660170</td>
      <td>0.604977</td>
      <td>0.637882</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>0.06055</td>
      <td>-0.515364</td>
      <td>-1.054615</td>
      <td>-1.062909</td>
      <td>0.128036</td>
      <td>-0.347450</td>
      <td>-0.961392</td>
      <td>0.611402</td>
      <td>0.137143</td>
      <td>-0.472719</td>
      <td>...</td>
      <td>-0.131356</td>
      <td>-0.400520</td>
      <td>0.015035</td>
      <td>-0.385179</td>
      <td>-0.285261</td>
      <td>-0.143364</td>
      <td>-0.288159</td>
      <td>-0.816464</td>
      <td>-0.786116</td>
      <td>-0.740253</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 31 columns</p>
</div>



Fit the model on the full training set, evaluate on test set:


```python
# Run this cell without changes
model.fit(X_train, y_train)
model.score(X_test, y_test)
```


```python
# __SOLUTION__
model.fit(X_train, y_train)
model.score(X_test, y_test)
```




    0.7943110080438888



Great, that worked! Now we have completed the full process of preprocessing the Ames Housing data in preparation for machine learning!

## Summary

In this cumulative lab, you used various techniques to prepare the Ames Housing data for modeling. You filtered down the full dataset to only relevant columns, filled in missing values, converted categorical data into numeric data, added interaction terms, and scaled the data. Each time, you practiced the scikit-learn transformer workflow by instantiating the transformer, fitting on the relevant training data, transforming the training data, and transforming the test data at the end (without re-instantiating or re-fitting the transformer object).
