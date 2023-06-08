for /L %%a in (1,1,10) do (

@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_2clients_100trees.yml

@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_2clients_10trees.yml

@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_2clients_100trees.yml

@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_5clients_10trees.yml

@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_5clients_100trees.yml

@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_5clients_1000trees.yml
@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_10clients_10trees.yml

@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_10clients_100trees.yml
@REM python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\config_10clients_100trees.yml

python .\F_RSF.v1\F_RSF\F_RSF_example.py -c .\F_RSF.v1\F_RSF\configs\baseline.yml
)