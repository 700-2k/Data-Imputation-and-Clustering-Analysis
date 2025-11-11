import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# 🌍 Globals
# ============================================


SMALL_PATH = 'data/small.xlsx'
MEDIUM_PATH = 'data/medium.xlsx'
LARGE_PATH = 'data/large.xlsx'

DELETED_SMALL_PATH = 'out/data_with_gaps/small'
DELETED_MEDIUM_PATH = 'out/data_with_gaps/medium'
DELETED_LARGE_PATH = 'out/data_with_gaps/large'

PERCENTS_OF_GAPS = [0.03, 0.05, 0.1, 0.2, 0.3]


# ============================================
# 📄 Dataset class
# ============================================


class Dataset():
    def __init__(self, path: str):
        self.df = self.load_data(path)       
        self.preprocess()
        
        
    #-------------Data Loading & Preprocess------------    
        
    def load_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(path)
        except FileNotFoundError:
            print(f"File {path} not exists")
            df = None
        return df
    
    
    def preprocess(self):
        self.df['date-time'] = pd.to_datetime(self.df['date-time'])
        self.df['cards_number'] = self.df['cards_number'].astype(str)
        

    #--------------Analysis & Diagrams---------------
    
    def count(self):
        df_count = pd.DataFrame()
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                df_count[col] = [self.df[col].mean(), self.df[col].median(), list(self.df[col].mode())[:5]]
            elif pd.api.types.is_string_dtype(self.df[col]):
                df_count[col] = [None, None, list(self.df[col].mode())[0]]
                
        return df_count
    
    
    def draw_hist_store(self):
        counts = self.df['store_name'].value_counts()

        plt.figure(figsize=(10, 8))
        plt.barh(counts.index, counts.values)
        plt.xlabel("Number of rows")
        plt.ylabel("Store Name")
        plt.tight_layout()
        plt.show()
        
        
    def draw_hist_date(self):
        self.df['date-time'] = pd.to_datetime(self.df['date-time'])

        df_month = pd.DataFrame()
        df_month['month'] = self.df['date-time'].dt.month

        def get_season(m):
            if m in [12, 1, 2]:
                return "Winter"
            elif m in [3, 4, 5]:
                return "Spring"
            elif m in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        df_month['season'] = df_month['month'].apply(get_season)

        plt.hist(df_month['season'])
        plt.xlabel("Season")
        plt.ylabel("Number of rows")
        plt.show()
        
        
    def draw_hist_coords(self):
        counts = self.df['coordinates'].value_counts()

        plt.figure(figsize=(12,5))
        plt.barh(counts.index, counts.values)
        plt.xlabel("Number of rows")
        plt.ylabel("Coordinates")
        plt.show()
        
        
    def draw_hist_cats(self):
        counts = self.df['categories'].value_counts()

        plt.figure(figsize=(10, 8))
        plt.barh(counts.index, counts.values)
        plt.xlabel("Number of rows")
        plt.ylabel("Category Name")
        plt.tight_layout()
        plt.show()
        
        
    def draw_hist_brand(self):
        counts = self.df['brands'].value_counts()
        
        plt.hist(counts, bins=30)
        plt.xlabel("Times the value occur")
        plt.ylabel("Number of values with this frequency")
        plt.title("Frequency distribution of categories")
        plt.show()
        
        
    def draw_hist_top_brands(self):
        counts = self.df['brands'].value_counts().head(10)
        
        plt.figure(figsize=(10,5))
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=45)
        plt.xlabel("Brands")
        plt.ylabel("Number of rows")
        plt.title("Top 10 Brands")
        plt.show()
        
        
    def draw_hist_bottom_brands(self):
        counts = self.df['brands'].value_counts().tail(10)
        
        plt.figure(figsize=(10,5))
        plt.bar(counts.index, counts.values)
        plt.xticks(rotation=45)
        plt.xlabel("Brands")
        plt.ylabel("Number of rows")
        plt.title("Top 10 Brands")
        plt.show()
        
        
    def draw_hist_price(self):
        plt.hist(self.df.loc[self.df['price'] <= 100000, 'price'], bins=50)
        plt.xlabel("Price")
        plt.ylabel("Number of rows")
        plt.show()
        
        
    def count_unique(self, feature: str):
        counts = self.df[feature].value_counts()
        return counts.describe()
    
    
    def draw_hist_num_products(self):
        counts = self.df['number_of_products'].value_counts().sort_index()
        counts.plot(kind='bar')
        plt.xlabel("Number_of_products")
        plt.ylabel("Number of rows")
        plt.xticks(rotation=0)
        plt.show()
        
        
    def analyse_receipt_id(self):
        print(f"Unique number: {self.df['receipt_id'].nunique()}, Number of rows: {len(self.df)}")
        print("Highest number of unique receipt id for stores:")
        print(self.df.groupby("store_name")["receipt_id"].nunique().head())
    
    
    def draw_hist_total_cost(self):
        plt.hist(self.df.loc[self.df['total_cost'] <= 100000, 'total_cost'], bins=50)
        plt.xlabel("Total cost")
        plt.ylabel("Number of rows")
        plt.show()
        
        
    #---------------------Make Gaps-----------------------
        
    def remove_blocks(self, percent: float=0.3, inplace: bool=True) -> pd.DataFrame:
        if inplace:
            df = self.df
        else:
            df = self.df.copy()
        
        rows, cols = df.shape
        total_cells = rows * cols
        target_remove = int(total_cells * percent)
        
        block_sizes = [(2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
        
        removed = 0
        
        while removed < target_remove:
            block_h, block_w = random.choice(block_sizes)
            
            r = random.randint(0, rows - block_h)
            c = random.randint(0, cols - block_w)
            
            for i in range(r, r + block_h):
                for j in range(c, c + block_w):
                    df.iat[i, j] = np.nan
                    
            removed += block_h * block_w
                    
        return df


# ============================================
# ⚙️ Global Functions
# ============================================


def create_data_with_gaps(data: dict[str, Dataset], percents: list[float]):
    for p in percents:
        deleted_small = data["small"].remove_blocks(percent=p, inplace=False)
        deleted_medium = data["medium"].remove_blocks(percent=p, inplace=False)
        deleted_large = data["large"].remove_blocks(percent=p, inplace=False)
        
        deleted_small.to_excel(f"{DELETED_SMALL_PATH}/{int(p*100)}.xlsx", index=False)
        deleted_medium.to_excel(f"{DELETED_MEDIUM_PATH}/{int(p*100)}.xlsx", index=False)
        deleted_large.to_excel(f"{DELETED_LARGE_PATH}/{int(p*100)}.xlsx", index=False)
        
        
# ============================================
# 📥 Loaded Data
# ============================================


data = {
    "small": Dataset(SMALL_PATH),
    "medium": Dataset(MEDIUM_PATH),
    "large": Dataset(LARGE_PATH),
}
