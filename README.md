# Book-Recommendations
## To run this project, you need to install
pip install jupyterlab pandas numpy matplotlib faiss-cpu
jupyter lab

## Structure
Data folder with gutenberg txt files
Src folder with notebooks
Src cache folder created automatically for saved artifacts

## Workflow
Notebook a clean tokenize
Remove gutenberg start and end boilerplate
Start at first preface or chapter line
Optional remove simple headings and transcriber notes
Lowercase and keep words only a to z
Produce tokens by book mapping in memory
Optional save tokens and counts to src cache

## Usage
Run it by command straight from the package
Can create new files to test out everything in JupyterLab as well

## Outputs
Tokens by book dictionary in memory
Counts parquet file in src cache

printed tables of top tf and top tf idf terms

nearest neighbor lists for selected books

topic ranking tables

optional figures saved from notebook