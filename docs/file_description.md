## Data
1. `all_data.csv`
    - Data dated 6 Jan 2020, downloaded from www.itjuzi.com
    - Contains all data after preliminary data cleansing, in particular, removing irrelevant attributes (e.g. company email, contact number) and overseas company

2. `company_news.csv`
    - News data dated 7 Jan 2020, scraped from www.itjuzi.com 

## Label
1. `id_label.json`
    - 'name':'company_id'; generated from scraped data

2. `founded_label.json`
    - 'company_id':'founded_date'; generated from scraped data


## Notebook
#### Data preparation:
1. `output_merge.ipynb`
    - Label window using funding event

2. `input_merge.ipynb`
    - Truncate all features into windows and concatenate 70 features into numpy.array in each window

#### Feature engineering:
3. `sentiment_news.ipynb` (news feature)

4. `prepare_features.ipynb` (static, funding and investor features)
    - Team profile attribute is merged from scraped data, thus the feature engineering code is omitted in this repo to reduce redundancy
    - GPS encoding of address using GoogleMap API is provided but this attribute is not used in model training due to proven worse model performance

5. `prepare_investor_input.ipynb` (investor feature)

