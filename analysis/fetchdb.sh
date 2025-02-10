#!/bin/sh

mongoexport --jsonArray --db riemannStudy --collection riemannCollection mongodb+srv://riemann-study.mhxq2bq.mongodb.net/ --username riemann-study123 --password SzGVFYvxfGVqy3lq -o "out.json"

python formatdb.py