## Base dependency 
text  Better ready-to-use text dataset built from EDGAR (too big we take some of them from the large part ):
https://huggingface.co/datasets/JanosAudran/financial-reports-sec

## Operational dependency text  these S&P 500 folder files :
https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts/tree/main/data
https://huggingface.co/datasets/kurry/sp500_earnings_transcripts/tree/main/parquet_files


pip install gdeltdoc
gdetdoc provides data about when an event happens , library is gdeltdoc and we can query it to give us the files we need instead of installing directly like others.


EDGAR gives you dependency/business text

earnings transcripts give you operational dependency text

gdeltdoc / GDELT gives you event text