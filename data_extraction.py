import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

url = 'https://www.bitcoin-otc.com/viewratings.php'
page = requests.get(url)

soup = BeautifulSoup(page.text, 'lxml')
table1 = soup.find("table", {"class":"datadisplay sortable"})

headers = []
for i in table1.find_all("th"):
 title = i.text
 headers.append(title)

print(headers)

first_row = table1.find_all("tr")[1]
first_tds = first_row.find_all("td")
first_url = 'https://www.bitcoin-otc.com/' + first_tds[1].a['href'];
first_page = requests.get(first_url)
first_soup = BeautifulSoup(first_page.text,'lxml')
first_table = first_soup.find("table", {"class":"datadisplay sortable"})

inner_headers = []
for i in first_table.find_all("th"):
  title = i.text
  inner_headers.append(title)

print(inner_headers)

node_data = pd.DataFrame(columns = headers)
network_data = pd.DataFrame(columns = inner_headers)


for j in tqdm(table1.find_all("tr")[1:]):
 row_data = j.find_all("td")
 row = [i.text for i in row_data]
 inner_url = 'https://www.bitcoin-otc.com/' + row_data[1].a['href']
 inner_page = requests.get(inner_url)
 inner_soup = BeautifulSoup(inner_page.text,'lxml')
 inner_table = inner_soup.find("table", {"class":"datadisplay sortable"})
 for k in inner_table.find_all("tr")[1:]:
   inner_row_data = k.find_all("td")
   #print(len(inner_row_data))
   inner_row = [l.text for l in inner_row_data]
   inner_length = len(network_data)
   network_data.loc[inner_length] = inner_row

 length = len(node_data)
 node_data.loc[length] = row

print(node_data.head(5))
print()
print(network_data.head(5))

node_data.to_csv("nodes.csv",index=False)
network_data.to_csv("network.csv",index=False)