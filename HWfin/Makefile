cv:
	python clone_detect.py cv

rv:
	python clone_detect.py rv

fv_ts:
	python clone_detect.py fv_ts

fv_tr:
	python clone_detect.py fv_tr $(N)

tr:
	python clone_detect.py tr $(N)

ts:
	python clone_detect.py ts

tar:
	python clone_detect.py tar

res:
	"C:\Program Files\Notepad++\notepad++.exe" dataset/result.csv

sql:
	@sqlite3 dataset/featvec.db

stat:
	@sqlite3 dataset/featvec.db "select type, count(*) from fvcache group by type;"

du:
	@du -h dataset/data.pkl
	@du -h dataset/featvec.db
	@du -h dataset/model.pkl
