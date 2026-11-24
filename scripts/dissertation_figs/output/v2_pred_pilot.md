# v2 prediction results: pilot

Run: `agera_out/pilot_a`; versions {'python': '3.10.12', 'platform': 'Linux-5.15.0-204-generic-x86_64-with-glibc2.35', 'numpy': '2.1.3', 'pandas': '2.2.3', 'sklearn': '1.5.2', 'torch': '2.3.0+cu121', 'xgboost': '2.1.3'}; seeds [0, 1]; tasks 224.

Prediction cells 23,688; error files 0.

Determinism: 224/224 prediction files bit-identical between pilot_a and pilot_b; differing 0 []; missing in B 0; HP tables differing 0.


## Main table: MPI time

| test_tier | socket | form | method | n_seeds | n_cells_mean | log_mae_mean | log_mae_sd | base_mae_mean | skill_mean | skill_sd | bias_mean | spearman_mean | fb105_mean | seeds_better_than_base | consistent_vs_base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | aware | 1w | ridge | 2 | 31.5000 | 0.1467 | 0.0147 | 0.1922 | 0.2379 | 0.0271 | -0.0396 | 0.5500 | 0.1053 | 2 | below |
| full | aware | 1w | lasso | 2 | 31.5000 | 0.1511 | 0.0184 | 0.1922 | 0.2153 | 0.0450 | -0.0321 | 0.4934 | 0.1579 | 2 | below |
| full | aware | 1w | svr | 2 | 31.5000 | 0.1198 | 0.0035 | 0.1922 | 0.3746 | 0.0590 | -0.0310 | 0.6398 | 0.1842 | 2 | below |
| full | aware | 1w | rf | 2 | 31.5000 | 0.1041 | 0.0203 | 0.1922 | 0.4608 | 0.0706 | -0.0314 | 0.6364 | 0.1842 | 2 | below |
| full | aware | 1w | xgb | 2 | 31.5000 | 0.1000 | 0.0223 | 0.1922 | 0.4824 | 0.0825 | -0.0168 | 0.6233 | 0.1842 | 2 | below |
| full | aware | 1w | mlp | 2 | 31.5000 | 0.1195 | 0.0037 | 0.1922 | 0.3779 | 0.0211 | 0.0195 | 0.4967 | 0.1316 | 2 | below |
| full | aware | bd | ridge | 2 | 31.5000 | 0.1446 | 0.0114 | 0.1922 | 0.2479 | 0.0103 | -0.0220 | 0.5228 | 0.1316 | 2 | below |
| full | aware | bd | lasso | 2 | 31.5000 | 0.1419 | 0.0099 | 0.1922 | 0.2620 | 0.0036 | -0.0292 | 0.5223 | 0.1316 | 2 | below |
| full | aware | bd | svr | 2 | 31.5000 | 0.1118 | 0.0195 | 0.1922 | 0.4207 | 0.0639 | -0.0120 | 0.5816 | 0.1316 | 2 | below |
| full | aware | bd | rf | 2 | 31.5000 | 0.1047 | 0.0213 | 0.1922 | 0.4580 | 0.0758 | -0.0290 | 0.6249 | 0.1842 | 2 | below |
| full | aware | bd | xgb | 2 | 31.5000 | 0.0922 | 0.0195 | 0.1922 | 0.5225 | 0.0703 | -0.0087 | 0.6462 | 0.1842 | 2 | below |
| full | aware | bd | mlp | 2 | 31.5000 | 0.0945 | 0.0265 | 0.1922 | 0.5118 | 0.1062 | 0.0108 | 0.6330 | 0.2895 | 2 | below |
| full | aware | none | absolute | 2 | 31.5000 | 0.1579 | 0.0296 | 0.1922 | 0.1718 | 0.2077 | 0.0328 | 0.3619 | 0.0000 | 2 | below |
| full | aware | none | delta | 2 | 31.5000 | 0.1464 | 0.0279 | 0.1922 | 0.2416 | 0.0960 | -0.0093 | 0.6114 | 0.2105 | 2 | below |
| full | blind | 1w | ridge | 2 | 31.5000 | 0.1466 | 0.0144 | 0.1922 | 0.2383 | 0.0256 | -0.0396 | 0.5127 | 0.1053 | 2 | below |
| full | blind | 1w | lasso | 2 | 31.5000 | 0.1511 | 0.0189 | 0.1922 | 0.2153 | 0.0475 | -0.0320 | 0.4799 | 0.1579 | 2 | below |
| full | blind | 1w | svr | 2 | 31.5000 | 0.1362 | 0.0114 | 0.1922 | 0.2917 | 0.0133 | -0.0522 | 0.5705 | 0.1316 | 2 | below |
| full | blind | 1w | rf | 2 | 31.5000 | 0.1075 | 0.0251 | 0.1922 | 0.4437 | 0.0943 | -0.0268 | 0.4987 | 0.1316 | 2 | below |
| full | blind | 1w | xgb | 2 | 31.5000 | 0.1136 | 0.0303 | 0.1922 | 0.4130 | 0.1197 | -0.0260 | 0.5064 | 0.1316 | 2 | below |
| full | blind | 1w | mlp | 2 | 31.5000 | 0.1155 | 0.0105 | 0.1922 | 0.3997 | 0.0157 | 0.0076 | 0.5339 | 0.2368 | 2 | below |
| full | blind | bd | ridge | 2 | 31.5000 | 0.1444 | 0.0113 | 0.1922 | 0.2491 | 0.0098 | -0.0220 | 0.5149 | 0.1842 | 2 | below |
| full | blind | bd | lasso | 2 | 31.5000 | 0.1420 | 0.0093 | 0.1922 | 0.2610 | 0.0003 | -0.0292 | 0.4957 | 0.0789 | 2 | below |
| full | blind | bd | svr | 2 | 31.5000 | 0.1123 | 0.0223 | 0.1922 | 0.4181 | 0.0780 | -0.0143 | 0.5893 | 0.1053 | 2 | below |
| full | blind | bd | rf | 2 | 31.5000 | 0.1089 | 0.0247 | 0.1922 | 0.4366 | 0.0919 | -0.0242 | 0.5372 | 0.1053 | 2 | below |
| full | blind | bd | xgb | 2 | 31.5000 | 0.1102 | 0.0231 | 0.1922 | 0.4291 | 0.0832 | -0.0235 | 0.5469 | 0.1053 | 2 | below |
| full | blind | bd | mlp | 2 | 31.5000 | 0.0966 | 0.0287 | 0.1922 | 0.5011 | 0.1168 | 0.0038 | 0.5462 | 0.1053 | 2 | below |
| full | blind | none | absolute | 2 | 31.5000 | 0.1505 | 0.0233 | 0.1922 | 0.2116 | 0.1722 | 0.0586 | 0.4458 | 0.1316 | 2 | below |
| full | blind | none | delta | 2 | 31.5000 | 0.1184 | 0.0415 | 0.1922 | 0.3897 | 0.1761 | 0.0087 | 0.5976 | 0.1579 | 2 | below |
| one_known | aware | 1w | ridge | 2 | 152.0000 | 0.1421 | 0.0000 | 0.1677 | 0.1529 | 0.0000 | -0.0716 | 0.2854 | 0.2826 | 2 | below |
| one_known | aware | 1w | lasso | 2 | 152.0000 | 0.1423 | 0.0000 | 0.1677 | 0.1514 | 0.0000 | -0.0685 | 0.1830 | 0.4457 | 2 | below |
| one_known | aware | 1w | svr | 2 | 152.0000 | 0.1455 | 0.0000 | 0.1677 | 0.1329 | 0.0000 | -0.0855 | 0.1599 | 0.5978 | 2 | below |
| one_known | aware | 1w | rf | 2 | 152.0000 | 0.1317 | 0.0003 | 0.1677 | 0.2150 | 0.0018 | -0.0560 | 0.2241 | 0.4457 | 2 | below |
| one_known | aware | 1w | xgb | 2 | 152.0000 | 0.1345 | 0.0008 | 0.1677 | 0.1980 | 0.0045 | -0.0601 | 0.3161 | 0.3859 | 2 | below |
| one_known | aware | 1w | mlp | 2 | 152.0000 | 0.1691 | 0.0107 | 0.1677 | -0.0078 | 0.0635 | -0.0743 | 0.3013 | 0.5000 | 1 | mixed |
| one_known | aware | bd | ridge | 2 | 152.0000 | 0.1327 | 0.0000 | 0.1677 | 0.2087 | 0.0000 | -0.0348 | 0.3667 | 0.1739 | 2 | below |
| one_known | aware | bd | lasso | 2 | 152.0000 | 0.1413 | 0.0000 | 0.1677 | 0.1577 | 0.0000 | -0.0695 | 0.2614 | 0.4783 | 2 | below |
| one_known | aware | bd | svr | 2 | 152.0000 | 0.1507 | 0.0000 | 0.1677 | 0.1015 | 0.0000 | -0.0936 | 0.1402 | 0.8152 | 2 | below |
| one_known | aware | bd | rf | 2 | 152.0000 | 0.1338 | 0.0004 | 0.1677 | 0.2026 | 0.0024 | -0.0709 | 0.3317 | 0.4891 | 2 | below |
| one_known | aware | bd | xgb | 2 | 152.0000 | 0.1313 | 0.0011 | 0.1677 | 0.2173 | 0.0066 | -0.0594 | 0.3785 | 0.3641 | 2 | below |
| one_known | aware | bd | mlp | 2 | 152.0000 | 0.2024 | 0.0273 | 0.1677 | -0.2064 | 0.1629 | 0.0803 | 0.3250 | 0.2880 | 0 | above |
| one_known | aware | none | absolute | 2 | 152.0000 | 0.2112 | 0.0037 | 0.1677 | -0.2588 | 0.0221 | 0.0104 | 0.2993 | 0.2500 | 0 | above |
| one_known | aware | none | delta | 2 | 152.0000 | 0.1384 | 0.0043 | 0.1677 | 0.1749 | 0.0258 | 0.0602 | 0.5363 | 0.1522 | 2 | below |
| one_known | blind | 1w | ridge | 2 | 152.0000 | 0.1424 | 0.0000 | 0.1677 | 0.1511 | 0.0000 | -0.0716 | 0.2586 | 0.3152 | 2 | below |
| one_known | blind | 1w | lasso | 2 | 152.0000 | 0.1430 | 0.0000 | 0.1677 | 0.1474 | 0.0000 | -0.0686 | 0.1337 | 0.5435 | 2 | below |
| one_known | blind | 1w | svr | 2 | 152.0000 | 0.1456 | 0.0000 | 0.1677 | 0.1321 | 0.0000 | -0.0863 | 0.1612 | 0.5870 | 2 | below |
| one_known | blind | 1w | rf | 2 | 152.0000 | 0.1356 | 0.0004 | 0.1677 | 0.1917 | 0.0021 | -0.0642 | 0.1984 | 0.4239 | 2 | below |
| one_known | blind | 1w | xgb | 2 | 152.0000 | 0.1367 | 0.0009 | 0.1677 | 0.1851 | 0.0056 | -0.0598 | 0.2193 | 0.4130 | 2 | below |
| one_known | blind | 1w | mlp | 2 | 152.0000 | 0.1623 | 0.0111 | 0.1677 | 0.0324 | 0.0660 | -0.0694 | 0.2253 | 0.4348 | 1 | mixed |
| one_known | blind | bd | ridge | 2 | 152.0000 | 0.1333 | 0.0000 | 0.1677 | 0.2054 | 0.0000 | -0.0348 | 0.3471 | 0.1848 | 2 | below |
| one_known | blind | bd | lasso | 2 | 152.0000 | 0.1425 | 0.0000 | 0.1677 | 0.1508 | 0.0000 | -0.0710 | 0.1858 | 0.5761 | 2 | below |
| one_known | blind | bd | svr | 2 | 152.0000 | 0.1557 | 0.0000 | 0.1677 | 0.0719 | 0.0000 | -0.0965 | 0.0647 | 0.8152 | 2 | below |
| one_known | blind | bd | rf | 2 | 152.0000 | 0.1373 | 0.0001 | 0.1677 | 0.1813 | 0.0004 | -0.0716 | 0.2098 | 0.5815 | 2 | below |
| one_known | blind | bd | xgb | 2 | 152.0000 | 0.1356 | 0.0006 | 0.1677 | 0.1914 | 0.0037 | -0.0658 | 0.3034 | 0.4239 | 2 | below |
| one_known | blind | bd | mlp | 2 | 152.0000 | 0.1537 | 0.0112 | 0.1677 | 0.0837 | 0.0670 | 0.0145 | 0.3222 | 0.3315 | 2 | below |
| one_known | blind | none | absolute | 2 | 152.0000 | 0.2048 | 0.0084 | 0.1677 | -0.2211 | 0.0503 | 0.1027 | 0.3609 | 0.1685 | 0 | above |
| one_known | blind | none | delta | 2 | 152.0000 | 0.1341 | 0.0018 | 0.1677 | 0.2005 | 0.0106 | 0.0269 | 0.5204 | 0.2011 | 2 | below |
| zero_shot | aware | 1w | ridge | 2 | 28.0000 | 0.1802 | 0.0000 | 0.2123 | 0.1514 | 0.0000 | -0.1179 | 0.2389 | 0.2381 | 2 | below |
| zero_shot | aware | 1w | lasso | 2 | 28.0000 | 0.1779 | 0.0000 | 0.2123 | 0.1623 | 0.0000 | -0.1133 | 0.3210 | 0.3333 | 2 | below |
| zero_shot | aware | 1w | svr | 2 | 28.0000 | 0.1878 | 0.0000 | 0.2123 | 0.1158 | 0.0000 | -0.1292 | 0.2220 | 0.4762 | 2 | below |
| zero_shot | aware | 1w | rf | 2 | 28.0000 | 0.1546 | 0.0012 | 0.2123 | 0.2721 | 0.0059 | -0.0773 | 0.3326 | 0.1905 | 2 | below |
| zero_shot | aware | 1w | xgb | 2 | 28.0000 | 0.1636 | 0.0005 | 0.2123 | 0.2296 | 0.0024 | -0.0942 | 0.4934 | 0.2143 | 2 | below |
| zero_shot | aware | 1w | mlp | 2 | 28.0000 | 0.2106 | 0.0155 | 0.2123 | 0.0080 | 0.0730 | -0.1356 | 0.3662 | 0.5000 | 1 | mixed |
| zero_shot | aware | bd | ridge | 2 | 28.0000 | 0.1597 | 0.0000 | 0.2123 | 0.2481 | 0.0000 | -0.0414 | 0.3684 | 0.0952 | 2 | below |
| zero_shot | aware | bd | lasso | 2 | 28.0000 | 0.1816 | 0.0000 | 0.2123 | 0.1447 | 0.0000 | -0.1146 | 0.3472 | 0.4286 | 2 | below |
| zero_shot | aware | bd | svr | 2 | 28.0000 | 0.1968 | 0.0000 | 0.2123 | 0.0733 | 0.0000 | -0.1447 | 0.1171 | 0.9524 | 2 | below |
| zero_shot | aware | bd | rf | 2 | 28.0000 | 0.1730 | 0.0006 | 0.2123 | 0.1855 | 0.0026 | -0.1158 | 0.4735 | 0.3095 | 2 | below |
| zero_shot | aware | bd | xgb | 2 | 28.0000 | 0.1604 | 0.0004 | 0.2123 | 0.2446 | 0.0020 | -0.0920 | 0.5764 | 0.1429 | 2 | below |
| zero_shot | aware | bd | mlp | 2 | 28.0000 | 0.2066 | 0.0518 | 0.2123 | 0.0270 | 0.2439 | 0.1034 | 0.3898 | 0.0952 | 1 | mixed |
| zero_shot | aware | none | absolute | 2 | 28.0000 | 0.3609 | 0.0245 | 0.2123 | -0.6995 | 0.1155 | 0.0363 | 0.0041 | 0.3810 | 0 | above |
| zero_shot | aware | none | delta | 2 | 28.0000 | 0.1338 | 0.0095 | 0.2123 | 0.3697 | 0.0449 | 0.0742 | 0.6685 | 0.0952 | 2 | below |
| zero_shot | blind | 1w | ridge | 2 | 28.0000 | 0.1806 | 0.0000 | 0.2123 | 0.1496 | 0.0000 | -0.1179 | 0.2237 | 0.1905 | 2 | below |
| zero_shot | blind | 1w | lasso | 2 | 28.0000 | 0.1792 | 0.0000 | 0.2123 | 0.1561 | 0.0000 | -0.1134 | 0.4873 | 0.3810 | 2 | below |
| zero_shot | blind | 1w | svr | 2 | 28.0000 | 0.1894 | 0.0000 | 0.2123 | 0.1079 | 0.0000 | -0.1318 | 0.3040 | 0.4286 | 2 | below |
| zero_shot | blind | 1w | rf | 2 | 28.0000 | 0.1610 | 0.0001 | 0.2123 | 0.2418 | 0.0006 | -0.0976 | 0.5898 | 0.1429 | 2 | below |
| zero_shot | blind | 1w | xgb | 2 | 28.0000 | 0.1635 | 0.0016 | 0.2123 | 0.2298 | 0.0075 | -0.0910 | 0.5819 | 0.0476 | 2 | below |
| zero_shot | blind | 1w | mlp | 2 | 28.0000 | 0.1928 | 0.0473 | 0.2123 | 0.0920 | 0.2228 | -0.1258 | 0.4300 | 0.4286 | 1 | mixed |
| zero_shot | blind | bd | ridge | 2 | 28.0000 | 0.1600 | 0.0000 | 0.2123 | 0.2463 | 0.0000 | -0.0414 | 0.3845 | 0.0476 | 2 | below |
| zero_shot | blind | bd | lasso | 2 | 28.0000 | 0.1830 | 0.0000 | 0.2123 | 0.1384 | 0.0000 | -0.1176 | 0.4196 | 0.5238 | 2 | below |
| zero_shot | blind | bd | svr | 2 | 28.0000 | 0.2010 | 0.0000 | 0.2123 | 0.0532 | 0.0000 | -0.1539 | 0.0156 | 0.9524 | 2 | below |
| zero_shot | blind | bd | rf | 2 | 28.0000 | 0.1767 | 0.0035 | 0.2123 | 0.1677 | 0.0166 | -0.1201 | 0.5724 | 0.5238 | 2 | below |
| zero_shot | blind | bd | xgb | 2 | 28.0000 | 0.1648 | 0.0004 | 0.2123 | 0.2241 | 0.0020 | -0.1002 | 0.5798 | 0.3571 | 2 | below |
| zero_shot | blind | bd | mlp | 2 | 28.0000 | 0.1454 | 0.0283 | 0.2123 | 0.3154 | 0.1331 | -0.0346 | 0.4999 | 0.1667 | 2 | below |
| zero_shot | blind | none | absolute | 2 | 28.0000 | 0.5790 | 0.2318 | 0.2123 | -1.7268 | 1.0918 | 0.2838 | 0.0889 | 0.2381 | 0 | above |
| zero_shot | blind | none | delta | 2 | 28.0000 | 0.1562 | 0.0381 | 0.2123 | 0.2642 | 0.1796 | 0.0495 | 0.5010 | 0.1667 | 2 | below |

## Main table: application time

| test_tier | socket | form | method | n_seeds | n_cells_mean | log_mae_mean | log_mae_sd | base_mae_mean | skill_mean | skill_sd | bias_mean | spearman_mean | fb105_mean | seeds_better_than_base | consistent_vs_base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | aware | 1w | ridge | 2 | 31.5000 | 0.0262 | 0.0032 | 0.0366 | 0.2865 | 0.0374 | -0.0055 | 0.7180 | 0.4833 | 2 | below |
| full | aware | 1w | lasso | 2 | 31.5000 | 0.0273 | 0.0039 | 0.0366 | 0.2561 | 0.0547 | -0.0083 | 0.6678 | 0.8167 | 2 | below |
| full | aware | 1w | svr | 2 | 31.5000 | 0.0221 | 0.0017 | 0.0366 | 0.3959 | 0.0043 | -0.0105 | 0.6840 | 0.7000 | 2 | below |
| full | aware | 1w | rf | 2 | 31.5000 | 0.0225 | 0.0015 | 0.0366 | 0.3849 | 0.0011 | -0.0100 | 0.8160 | 0.5333 | 2 | below |
| full | aware | 1w | xgb | 2 | 31.5000 | 0.0209 | 0.0010 | 0.0366 | 0.4291 | 0.0134 | -0.0090 | 0.7779 | 0.4500 | 2 | below |
| full | aware | 1w | mlp | 2 | 31.5000 | 0.0214 | 0.0013 | 0.0366 | 0.4144 | 0.0064 | 0.0002 | 0.6959 | 0.2833 | 2 | below |
| full | aware | bd | ridge | 2 | 31.5000 | 0.0283 | 0.0008 | 0.0366 | 0.2262 | 0.0322 | -0.0030 | 0.7269 | 0.4833 | 2 | below |
| full | aware | bd | lasso | 2 | 31.5000 | 0.0279 | 0.0019 | 0.0366 | 0.2367 | 0.0003 | -0.0057 | 0.6893 | 0.7333 | 2 | below |
| full | aware | bd | svr | 2 | 31.5000 | 0.0207 | 0.0048 | 0.0366 | 0.4379 | 0.0920 | -0.0078 | 0.7706 | 0.4500 | 2 | below |
| full | aware | bd | rf | 2 | 31.5000 | 0.0207 | 0.0027 | 0.0366 | 0.4352 | 0.0332 | -0.0119 | 0.8398 | 0.4500 | 2 | below |
| full | aware | bd | xgb | 2 | 31.5000 | 0.0225 | 0.0011 | 0.0366 | 0.3847 | 0.0121 | -0.0071 | 0.7472 | 0.3667 | 2 | below |
| full | aware | bd | mlp | 2 | 31.5000 | 0.0187 | 0.0048 | 0.0366 | 0.4921 | 0.0963 | -0.0008 | 0.7364 | 0.2833 | 2 | below |
| full | aware | none | absolute | 2 | 31.5000 | 0.0382 | 0.0042 | 0.0366 | -0.0427 | 0.0431 | 0.0005 | 0.1251 | 0.6500 | 0 | above |
| full | aware | none | delta | 2 | 31.5000 | 0.0215 | 0.0017 | 0.0366 | 0.4133 | 0.0060 | -0.0008 | 0.7440 | 0.2000 | 2 | below |
| full | blind | 1w | ridge | 2 | 31.5000 | 0.0262 | 0.0032 | 0.0366 | 0.2856 | 0.0378 | -0.0055 | 0.7138 | 0.4833 | 2 | below |
| full | blind | 1w | lasso | 2 | 31.5000 | 0.0273 | 0.0039 | 0.0366 | 0.2561 | 0.0547 | -0.0083 | 0.6678 | 0.8167 | 2 | below |
| full | blind | 1w | svr | 2 | 31.5000 | 0.0215 | 0.0012 | 0.0366 | 0.4119 | 0.0078 | -0.0105 | 0.7291 | 0.7000 | 2 | below |
| full | blind | 1w | rf | 2 | 31.5000 | 0.0226 | 0.0020 | 0.0366 | 0.3827 | 0.0116 | -0.0101 | 0.8059 | 0.5333 | 2 | below |
| full | blind | 1w | xgb | 2 | 31.5000 | 0.0212 | 0.0006 | 0.0366 | 0.4213 | 0.0231 | -0.0081 | 0.7995 | 0.3667 | 2 | below |
| full | blind | 1w | mlp | 2 | 31.5000 | 0.0215 | 0.0006 | 0.0366 | 0.4117 | 0.0257 | -0.0011 | 0.6816 | 0.2833 | 2 | below |
| full | blind | bd | ridge | 2 | 31.5000 | 0.0283 | 0.0008 | 0.0366 | 0.2254 | 0.0316 | -0.0030 | 0.7336 | 0.4000 | 2 | below |
| full | blind | bd | lasso | 2 | 31.5000 | 0.0279 | 0.0019 | 0.0366 | 0.2367 | 0.0003 | -0.0057 | 0.6893 | 0.7333 | 2 | below |
| full | blind | bd | svr | 2 | 31.5000 | 0.0217 | 0.0023 | 0.0366 | 0.4086 | 0.0224 | -0.0105 | 0.7627 | 0.7000 | 2 | below |
| full | blind | bd | rf | 2 | 31.5000 | 0.0224 | 0.0031 | 0.0366 | 0.3888 | 0.0416 | -0.0125 | 0.8284 | 0.6500 | 2 | below |
| full | blind | bd | xgb | 2 | 31.5000 | 0.0209 | 0.0000 | 0.0366 | 0.4264 | 0.0397 | -0.0080 | 0.8383 | 0.3667 | 2 | below |
| full | blind | bd | mlp | 2 | 31.5000 | 0.0197 | 0.0061 | 0.0366 | 0.4672 | 0.1301 | -0.0004 | 0.6856 | 0.2000 | 2 | below |
| full | blind | none | absolute | 2 | 31.5000 | 0.0398 | 0.0007 | 0.0366 | -0.0918 | 0.0952 | -0.0012 | 0.1756 | 0.7500 | 0 | above |
| full | blind | none | delta | 2 | 31.5000 | 0.0232 | 0.0089 | 0.0366 | 0.3728 | 0.1987 | 0.0018 | 0.7616 | 0.2833 | 2 | below |
| one_known | aware | 1w | ridge | 2 | 152.0000 | 0.0181 | 0.0000 | 0.0227 | 0.2009 | 0.0000 | -0.0101 | 0.6311 | 0.8000 | 2 | below |
| one_known | aware | 1w | lasso | 2 | 152.0000 | 0.0200 | 0.0000 | 0.0227 | 0.1202 | 0.0000 | -0.0116 | 0.3804 | 1.0000 | 2 | below |
| one_known | aware | 1w | svr | 2 | 152.0000 | 0.0193 | 0.0000 | 0.0227 | 0.1507 | 0.0000 | -0.0142 | 0.4699 | 1.0000 | 2 | below |
| one_known | aware | 1w | rf | 2 | 152.0000 | 0.0167 | 0.0001 | 0.0227 | 0.2666 | 0.0031 | -0.0077 | 0.6799 | 0.6667 | 2 | below |
| one_known | aware | 1w | xgb | 2 | 152.0000 | 0.0166 | 0.0006 | 0.0227 | 0.2702 | 0.0266 | -0.0093 | 0.7114 | 0.7333 | 2 | below |
| one_known | aware | 1w | mlp | 2 | 152.0000 | 0.0186 | 0.0004 | 0.0227 | 0.1827 | 0.0191 | -0.0058 | 0.6047 | 0.7333 | 2 | below |
| one_known | aware | bd | ridge | 2 | 152.0000 | 0.0217 | 0.0000 | 0.0227 | 0.0451 | 0.0000 | -0.0067 | 0.2268 | 0.8000 | 2 | below |
| one_known | aware | bd | lasso | 2 | 152.0000 | 0.0211 | 0.0000 | 0.0227 | 0.0707 | 0.0000 | -0.0128 | 0.1106 | 1.0000 | 2 | below |
| one_known | aware | bd | svr | 2 | 152.0000 | 0.0201 | 0.0000 | 0.0227 | 0.1131 | 0.0000 | -0.0148 | 0.4827 | 1.0000 | 2 | below |
| one_known | aware | bd | rf | 2 | 152.0000 | 0.0178 | 0.0001 | 0.0227 | 0.2142 | 0.0045 | -0.0106 | 0.5595 | 0.7333 | 2 | below |
| one_known | aware | bd | xgb | 2 | 152.0000 | 0.0166 | 0.0000 | 0.0227 | 0.2703 | 0.0009 | -0.0098 | 0.6364 | 0.8000 | 2 | below |
| one_known | aware | bd | mlp | 2 | 152.0000 | 0.0205 | 0.0013 | 0.0227 | 0.0969 | 0.0571 | 0.0009 | 0.4840 | 0.3333 | 2 | below |
| one_known | aware | none | absolute | 2 | 152.0000 | 0.0254 | 0.0016 | 0.0227 | -0.1165 | 0.0708 | 0.0063 | 0.1047 | 0.8000 | 0 | above |
| one_known | aware | none | delta | 2 | 152.0000 | 0.0229 | 0.0018 | 0.0227 | -0.0092 | 0.0789 | -0.0026 | 0.6206 | 0.4333 | 1 | mixed |
| one_known | blind | 1w | ridge | 2 | 152.0000 | 0.0182 | 0.0000 | 0.0227 | 0.1987 | 0.0000 | -0.0101 | 0.6270 | 0.8000 | 2 | below |
| one_known | blind | 1w | lasso | 2 | 152.0000 | 0.0200 | 0.0000 | 0.0227 | 0.1202 | 0.0000 | -0.0116 | 0.3804 | 1.0000 | 2 | below |
| one_known | blind | 1w | svr | 2 | 152.0000 | 0.0194 | 0.0000 | 0.0227 | 0.1457 | 0.0000 | -0.0138 | 0.5095 | 1.0000 | 2 | below |
| one_known | blind | 1w | rf | 2 | 152.0000 | 0.0169 | 0.0000 | 0.0227 | 0.2551 | 0.0021 | -0.0076 | 0.6691 | 0.6667 | 2 | below |
| one_known | blind | 1w | xgb | 2 | 152.0000 | 0.0165 | 0.0008 | 0.0227 | 0.2730 | 0.0334 | -0.0091 | 0.6995 | 0.7000 | 2 | below |
| one_known | blind | 1w | mlp | 2 | 152.0000 | 0.0185 | 0.0014 | 0.0227 | 0.1857 | 0.0610 | -0.0063 | 0.6418 | 0.7333 | 2 | below |
| one_known | blind | bd | ridge | 2 | 152.0000 | 0.0218 | 0.0000 | 0.0227 | 0.0418 | 0.0000 | -0.0067 | 0.2221 | 0.8000 | 2 | below |
| one_known | blind | bd | lasso | 2 | 152.0000 | 0.0211 | 0.0000 | 0.0227 | 0.0707 | 0.0000 | -0.0128 | 0.1106 | 1.0000 | 2 | below |
| one_known | blind | bd | svr | 2 | 152.0000 | 0.0198 | 0.0000 | 0.0227 | 0.1302 | 0.0000 | -0.0139 | 0.4290 | 1.0000 | 2 | below |
| one_known | blind | bd | rf | 2 | 152.0000 | 0.0176 | 0.0000 | 0.0227 | 0.2238 | 0.0020 | -0.0115 | 0.5564 | 0.8000 | 2 | below |
| one_known | blind | bd | xgb | 2 | 152.0000 | 0.0166 | 0.0003 | 0.0227 | 0.2675 | 0.0113 | -0.0097 | 0.6290 | 0.8000 | 2 | below |
| one_known | blind | bd | mlp | 2 | 152.0000 | 0.0201 | 0.0046 | 0.0227 | 0.1150 | 0.2032 | 0.0004 | 0.4440 | 0.3667 | 1 | mixed |
| one_known | blind | none | absolute | 2 | 152.0000 | 0.0259 | 0.0052 | 0.0227 | -0.1421 | 0.2295 | 0.0078 | 0.1639 | 0.8333 | 1 | mixed |
| one_known | blind | none | delta | 2 | 152.0000 | 0.0187 | 0.0009 | 0.0227 | 0.1744 | 0.0402 | -0.0001 | 0.6476 | 0.4333 | 2 | below |
| zero_shot | aware | 1w | ridge | 2 | 28.0000 | 0.0398 | 0.0000 | 0.0450 | 0.1159 | 0.0000 | -0.0325 | 0.4963 | 0.8571 | 2 | below |
| zero_shot | aware | 1w | lasso | 2 | 28.0000 | 0.0424 | 0.0000 | 0.0450 | 0.0570 | 0.0000 | -0.0350 | 0.1580 | 1.0000 | 2 | below |
| zero_shot | aware | 1w | svr | 2 | 28.0000 | 0.0409 | 0.0000 | 0.0450 | 0.0918 | 0.0000 | -0.0363 | 0.1213 | 1.0000 | 2 | below |
| zero_shot | aware | 1w | rf | 2 | 28.0000 | 0.0382 | 0.0005 | 0.0450 | 0.1508 | 0.0116 | -0.0280 | 0.5567 | 0.8571 | 2 | below |
| zero_shot | aware | 1w | xgb | 2 | 28.0000 | 0.0364 | 0.0001 | 0.0450 | 0.1903 | 0.0019 | -0.0326 | 0.8323 | 0.8571 | 2 | below |
| zero_shot | aware | 1w | mlp | 2 | 28.0000 | 0.0382 | 0.0007 | 0.0450 | 0.1517 | 0.0152 | -0.0265 | 0.5623 | 0.8571 | 2 | below |
| zero_shot | aware | bd | ridge | 2 | 28.0000 | 0.0403 | 0.0000 | 0.0450 | 0.1044 | 0.0000 | -0.0252 | 0.2644 | 1.0000 | 2 | below |
| zero_shot | aware | bd | lasso | 2 | 28.0000 | 0.0443 | 0.0000 | 0.0450 | 0.0151 | 0.0000 | -0.0377 | -0.1944 | 1.0000 | 2 | below |
| zero_shot | aware | bd | svr | 2 | 28.0000 | 0.0428 | 0.0000 | 0.0450 | 0.0480 | 0.0000 | -0.0370 | 0.3786 | 1.0000 | 2 | below |
| zero_shot | aware | bd | rf | 2 | 28.0000 | 0.0399 | 0.0001 | 0.0450 | 0.1133 | 0.0026 | -0.0341 | 0.2599 | 1.0000 | 2 | below |
| zero_shot | aware | bd | xgb | 2 | 28.0000 | 0.0401 | 0.0002 | 0.0450 | 0.1097 | 0.0046 | -0.0322 | 0.2430 | 0.9286 | 2 | below |
| zero_shot | aware | bd | mlp | 2 | 28.0000 | 0.0373 | 0.0026 | 0.0450 | 0.1719 | 0.0576 | -0.0227 | 0.4217 | 0.5714 | 2 | below |
| zero_shot | aware | none | absolute | 2 | 28.0000 | 0.0424 | 0.0001 | 0.0450 | 0.0568 | 0.0018 | -0.0166 | -0.1116 | 0.9286 | 2 | below |
| zero_shot | aware | none | delta | 2 | 28.0000 | 0.0198 | 0.0000 | 0.0450 | 0.5592 | 0.0006 | 0.0008 | 0.8642 | 0.1429 | 2 | below |
| zero_shot | blind | 1w | ridge | 2 | 28.0000 | 0.0399 | 0.0000 | 0.0450 | 0.1130 | 0.0000 | -0.0325 | 0.4964 | 0.8571 | 2 | below |
| zero_shot | blind | 1w | lasso | 2 | 28.0000 | 0.0424 | 0.0000 | 0.0450 | 0.0570 | 0.0000 | -0.0350 | 0.1580 | 1.0000 | 2 | below |
| zero_shot | blind | 1w | svr | 2 | 28.0000 | 0.0406 | 0.0000 | 0.0450 | 0.0981 | 0.0000 | -0.0358 | 0.3122 | 1.0000 | 2 | below |
| zero_shot | blind | 1w | rf | 2 | 28.0000 | 0.0384 | 0.0002 | 0.0450 | 0.1461 | 0.0044 | -0.0281 | 0.5476 | 0.8571 | 2 | below |
| zero_shot | blind | 1w | xgb | 2 | 28.0000 | 0.0356 | 0.0003 | 0.0450 | 0.2086 | 0.0059 | -0.0307 | 0.7741 | 0.8571 | 2 | below |
| zero_shot | blind | 1w | mlp | 2 | 28.0000 | 0.0368 | 0.0024 | 0.0450 | 0.1822 | 0.0535 | -0.0269 | 0.7086 | 0.8571 | 2 | below |
| zero_shot | blind | bd | ridge | 2 | 28.0000 | 0.0403 | 0.0000 | 0.0450 | 0.1031 | 0.0000 | -0.0252 | 0.2533 | 1.0000 | 2 | below |
| zero_shot | blind | bd | lasso | 2 | 28.0000 | 0.0443 | 0.0000 | 0.0450 | 0.0151 | 0.0000 | -0.0377 | -0.1944 | 1.0000 | 2 | below |
| zero_shot | blind | bd | svr | 2 | 28.0000 | 0.0420 | 0.0000 | 0.0450 | 0.0654 | 0.0000 | -0.0360 | 0.1236 | 1.0000 | 2 | below |
| zero_shot | blind | bd | rf | 2 | 28.0000 | 0.0411 | 0.0002 | 0.0450 | 0.0873 | 0.0037 | -0.0350 | 0.2973 | 1.0000 | 2 | below |
| zero_shot | blind | bd | xgb | 2 | 28.0000 | 0.0402 | 0.0005 | 0.0450 | 0.1058 | 0.0111 | -0.0322 | 0.2124 | 0.9286 | 2 | below |
| zero_shot | blind | bd | mlp | 2 | 28.0000 | 0.0355 | 0.0028 | 0.0450 | 0.2106 | 0.0633 | -0.0215 | 0.4519 | 0.6429 | 2 | below |
| zero_shot | blind | none | absolute | 2 | 28.0000 | 0.0423 | 0.0000 | 0.0450 | 0.0599 | 0.0005 | -0.0166 | -0.0612 | 0.7857 | 2 | below |
| zero_shot | blind | none | delta | 2 | 28.0000 | 0.0166 | 0.0007 | 0.0450 | 0.6301 | 0.0160 | -0.0015 | 0.8865 | 0.1429 | 2 | below |

## Sensitivity without self-pairs (skill mean)

| target | test_tier | socket | form | absolute | delta | lasso | mlp | rf | ridge | svr | xgb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpi | full | aware | 1w | nan | nan | 0.1792 | 0.3915 | 0.4528 | 0.2080 | 0.3768 | 0.4804 |
| mpi | full | aware | bd | nan | nan | 0.2351 | 0.5408 | 0.4467 | 0.2111 | 0.4202 | 0.5189 |
| mpi | full | aware | none | 0.1411 | 0.2239 | nan | nan | nan | nan | nan | nan |
| mpi | full | blind | 1w | nan | nan | 0.1793 | 0.3932 | 0.4402 | 0.2085 | 0.2855 | 0.4033 |
| mpi | full | blind | bd | nan | nan | 0.2342 | 0.5388 | 0.4278 | 0.2125 | 0.4130 | 0.4205 |
| mpi | full | blind | none | 0.1859 | 0.4048 | nan | nan | nan | nan | nan | nan |
| mpi | one_known | aware | 1w | nan | nan | 0.1514 | -0.0078 | 0.2150 | 0.1529 | 0.1329 | 0.1980 |
| mpi | one_known | aware | bd | nan | nan | 0.1577 | -0.2064 | 0.2026 | 0.2087 | 0.1015 | 0.2173 |
| mpi | one_known | aware | none | -0.2588 | 0.1749 | nan | nan | nan | nan | nan | nan |
| mpi | one_known | blind | 1w | nan | nan | 0.1474 | 0.0324 | 0.1917 | 0.1511 | 0.1321 | 0.1851 |
| mpi | one_known | blind | bd | nan | nan | 0.1508 | 0.0837 | 0.1813 | 0.2054 | 0.0719 | 0.1914 |
| mpi | one_known | blind | none | -0.2211 | 0.2005 | nan | nan | nan | nan | nan | nan |
| mpi | zero_shot | aware | 1w | nan | nan | 0.1790 | -0.0249 | 0.2873 | 0.1866 | 0.1310 | 0.2435 |
| mpi | zero_shot | aware | bd | nan | nan | 0.1500 | -0.2004 | 0.1918 | 0.2217 | 0.0854 | 0.2408 |
| mpi | zero_shot | aware | none | -0.8214 | 0.2600 | nan | nan | nan | nan | nan | nan |
| mpi | zero_shot | blind | 1w | nan | nan | 0.1700 | 0.0609 | 0.2578 | 0.1840 | 0.1173 | 0.2441 |
| mpi | zero_shot | blind | bd | nan | nan | 0.1454 | 0.2829 | 0.1755 | 0.2191 | 0.0610 | 0.2166 |
| mpi | zero_shot | blind | none | -1.7251 | 0.1308 | nan | nan | nan | nan | nan | nan |
| tt | full | aware | 1w | nan | nan | 0.2500 | 0.4101 | 0.3549 | 0.2690 | 0.4146 | 0.4295 |
| tt | full | aware | bd | nan | nan | 0.2279 | 0.4872 | 0.4424 | 0.2014 | 0.4413 | 0.3697 |
| tt | full | aware | none | -0.0517 | 0.4045 | nan | nan | nan | nan | nan | nan |
| tt | full | blind | 1w | nan | nan | 0.2500 | 0.4034 | 0.3555 | 0.2679 | 0.4340 | 0.4197 |
| tt | full | blind | bd | nan | nan | 0.2279 | 0.4604 | 0.3787 | 0.2004 | 0.4252 | 0.4235 |
| tt | full | blind | none | -0.1004 | 0.3954 | nan | nan | nan | nan | nan | nan |
| tt | one_known | aware | 1w | nan | nan | 0.1202 | 0.1827 | 0.2666 | 0.2009 | 0.1507 | 0.2702 |
| tt | one_known | aware | bd | nan | nan | 0.0707 | 0.0969 | 0.2142 | 0.0451 | 0.1131 | 0.2703 |
| tt | one_known | aware | none | -0.1165 | -0.0092 | nan | nan | nan | nan | nan | nan |
| tt | one_known | blind | 1w | nan | nan | 0.1202 | 0.1857 | 0.2551 | 0.1987 | 0.1457 | 0.2730 |
| tt | one_known | blind | bd | nan | nan | 0.0707 | 0.1150 | 0.2238 | 0.0418 | 0.1302 | 0.2675 |
| tt | one_known | blind | none | -0.1421 | 0.1744 | nan | nan | nan | nan | nan | nan |
| tt | zero_shot | aware | 1w | nan | nan | 0.0713 | 0.1447 | 0.1427 | 0.1101 | 0.1085 | 0.1877 |
| tt | zero_shot | aware | bd | nan | nan | 0.0095 | 0.1173 | 0.1224 | 0.0627 | 0.0549 | 0.1022 |
| tt | zero_shot | aware | none | 0.0791 | 0.4988 | nan | nan | nan | nan | nan | nan |
| tt | zero_shot | blind | 1w | nan | nan | 0.0713 | 0.1705 | 0.1351 | 0.1052 | 0.1109 | 0.2024 |
| tt | zero_shot | blind | bd | nan | nan | 0.0095 | 0.1555 | 0.0878 | 0.0606 | 0.0720 | 0.0967 |
| tt | zero_shot | blind | none | 0.0638 | 0.6274 | nan | nan | nan | nan | nan | nan |

## Paired contrasts (per-seed pair-cluster bootstrap)

| contrast | target | test_tier | socket | form | method | n_seeds | diff_mean | diff_sd | seeds_ci_below0 | seeds_ci_above0 | consistent_sign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aware-blind | mpi | full | aware-blind | 1w | lasso | 2 | -0.0000 | 0.0005 | 0 | 0 | mixed |
| aware-blind | mpi | full | aware-blind | 1w | mlp | 2 | 0.0040 | 0.0068 | 0 | 0 | mixed |
| aware-blind | mpi | full | aware-blind | 1w | rf | 2 | -0.0034 | 0.0048 | 0 | 0 | neg |
| aware-blind | mpi | full | aware-blind | 1w | ridge | 2 | 0.0001 | 0.0003 | 0 | 0 | mixed |
| aware-blind | mpi | full | aware-blind | 1w | svr | 2 | -0.0164 | 0.0149 | 1 | 0 | neg |
| aware-blind | mpi | full | aware-blind | 1w | xgb | 2 | -0.0136 | 0.0080 | 0 | 0 | neg |
| aware-blind | mpi | full | aware-blind | bd | lasso | 2 | -0.0002 | 0.0006 | 0 | 0 | mixed |
| aware-blind | mpi | full | aware-blind | bd | mlp | 2 | -0.0021 | 0.0022 | 0 | 0 | neg |
| aware-blind | mpi | full | aware-blind | bd | rf | 2 | -0.0042 | 0.0034 | 0 | 0 | neg |
| aware-blind | mpi | full | aware-blind | bd | ridge | 2 | 0.0002 | 0.0001 | 0 | 0 | pos |
| aware-blind | mpi | full | aware-blind | bd | svr | 2 | -0.0006 | 0.0027 | 0 | 0 | mixed |
| aware-blind | mpi | full | aware-blind | bd | xgb | 2 | -0.0180 | 0.0037 | 0 | 0 | neg |
| aware-blind | mpi | full | aware-blind | none | absolute | 2 | 0.0074 | 0.0063 | 0 | 0 | pos |
| aware-blind | mpi | full | aware-blind | none | delta | 2 | 0.0280 | 0.0135 | 0 | 1 | pos |
| aware-blind | mpi | one_known | aware-blind | 1w | lasso | 2 | -0.0007 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | 1w | mlp | 2 | 0.0067 | 0.0004 | 0 | 0 | pos |
| aware-blind | mpi | one_known | aware-blind | 1w | rf | 2 | -0.0039 | 0.0007 | 0 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | 1w | ridge | 2 | -0.0003 | 0.0000 | 2 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | 1w | svr | 2 | -0.0001 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | 1w | xgb | 2 | -0.0022 | 0.0002 | 0 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | bd | lasso | 2 | -0.0012 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | bd | mlp | 2 | 0.0487 | 0.0386 | 0 | 2 | pos |
| aware-blind | mpi | one_known | aware-blind | bd | rf | 2 | -0.0036 | 0.0003 | 1 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | bd | ridge | 2 | -0.0006 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | bd | svr | 2 | -0.0050 | 0.0000 | 2 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | bd | xgb | 2 | -0.0043 | 0.0005 | 0 | 0 | neg |
| aware-blind | mpi | one_known | aware-blind | none | absolute | 2 | 0.0063 | 0.0047 | 0 | 0 | pos |
| aware-blind | mpi | one_known | aware-blind | none | delta | 2 | 0.0043 | 0.0025 | 0 | 0 | pos |
| aware-blind | mpi | zero_shot | aware-blind | 1w | lasso | 2 | -0.0013 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | 1w | mlp | 2 | 0.0178 | 0.0318 | 0 | 1 | mixed |
| aware-blind | mpi | zero_shot | aware-blind | 1w | rf | 2 | -0.0064 | 0.0011 | 0 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | 1w | ridge | 2 | -0.0004 | 0.0000 | 2 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | 1w | svr | 2 | -0.0017 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | 1w | xgb | 2 | 0.0000 | 0.0011 | 0 | 0 | mixed |
| aware-blind | mpi | zero_shot | aware-blind | bd | lasso | 2 | -0.0013 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | bd | mlp | 2 | 0.0612 | 0.0235 | 0 | 0 | pos |
| aware-blind | mpi | zero_shot | aware-blind | bd | rf | 2 | -0.0038 | 0.0030 | 1 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | bd | ridge | 2 | -0.0004 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | bd | svr | 2 | -0.0043 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | bd | xgb | 2 | -0.0044 | 0.0000 | 0 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | none | absolute | 2 | -0.2182 | 0.2564 | 1 | 0 | neg |
| aware-blind | mpi | zero_shot | aware-blind | none | delta | 2 | -0.0224 | 0.0477 | 1 | 0 | mixed |
| aware-blind | tt | full | aware-blind | 1w | lasso | 2 | 0.0000 | 0.0000 | 0 | 0 | mixed |
| aware-blind | tt | full | aware-blind | 1w | mlp | 2 | -0.0001 | 0.0007 | 0 | 0 | mixed |
| aware-blind | tt | full | aware-blind | 1w | rf | 2 | -0.0001 | 0.0005 | 0 | 0 | mixed |
| aware-blind | tt | full | aware-blind | 1w | ridge | 2 | -0.0000 | 0.0000 | 0 | 0 | neg |
| aware-blind | tt | full | aware-blind | 1w | svr | 2 | 0.0006 | 0.0005 | 0 | 0 | pos |
| aware-blind | tt | full | aware-blind | 1w | xgb | 2 | -0.0003 | 0.0003 | 0 | 0 | neg |
| aware-blind | tt | full | aware-blind | bd | lasso | 2 | 0.0000 | 0.0000 | 0 | 0 | mixed |
| aware-blind | tt | full | aware-blind | bd | mlp | 2 | -0.0010 | 0.0013 | 0 | 0 | neg |
| aware-blind | tt | full | aware-blind | bd | rf | 2 | -0.0017 | 0.0004 | 0 | 0 | neg |
| aware-blind | tt | full | aware-blind | bd | ridge | 2 | -0.0000 | 0.0000 | 0 | 0 | neg |
| aware-blind | tt | full | aware-blind | bd | svr | 2 | -0.0010 | 0.0025 | 0 | 0 | mixed |
| aware-blind | tt | full | aware-blind | bd | xgb | 2 | 0.0016 | 0.0011 | 0 | 1 | pos |
| aware-blind | tt | full | aware-blind | none | absolute | 2 | -0.0016 | 0.0049 | 0 | 0 | mixed |
| aware-blind | tt | full | aware-blind | none | delta | 2 | -0.0017 | 0.0072 | 0 | 0 | mixed |
| aware-blind | tt | one_known | aware-blind | 1w | lasso | 2 | 0.0000 | 0.0000 | 0 | 0 | mixed |
| aware-blind | tt | one_known | aware-blind | 1w | mlp | 2 | 0.0001 | 0.0010 | 0 | 0 | mixed |
| aware-blind | tt | one_known | aware-blind | 1w | rf | 2 | -0.0003 | 0.0001 | 1 | 0 | neg |
| aware-blind | tt | one_known | aware-blind | 1w | ridge | 2 | -0.0001 | 0.0000 | 0 | 0 | neg |
| aware-blind | tt | one_known | aware-blind | 1w | svr | 2 | -0.0001 | 0.0000 | 0 | 0 | neg |
| aware-blind | tt | one_known | aware-blind | 1w | xgb | 2 | 0.0001 | 0.0002 | 0 | 0 | mixed |
| aware-blind | tt | one_known | aware-blind | bd | lasso | 2 | 0.0000 | 0.0000 | 0 | 0 | mixed |
| aware-blind | tt | one_known | aware-blind | bd | mlp | 2 | 0.0004 | 0.0033 | 1 | 1 | mixed |
| aware-blind | tt | one_known | aware-blind | bd | rf | 2 | 0.0002 | 0.0001 | 0 | 0 | pos |
| aware-blind | tt | one_known | aware-blind | bd | ridge | 2 | -0.0001 | 0.0000 | 2 | 0 | neg |
| aware-blind | tt | one_known | aware-blind | bd | svr | 2 | 0.0004 | 0.0000 | 0 | 0 | pos |
| aware-blind | tt | one_known | aware-blind | bd | xgb | 2 | -0.0001 | 0.0002 | 1 | 0 | mixed |
| aware-blind | tt | one_known | aware-blind | none | absolute | 2 | -0.0006 | 0.0036 | 1 | 1 | mixed |
| aware-blind | tt | one_known | aware-blind | none | delta | 2 | 0.0042 | 0.0009 | 0 | 1 | pos |
| aware-blind | tt | zero_shot | aware-blind | 1w | lasso | 2 | 0.0000 | 0.0000 | 0 | 0 | mixed |
| aware-blind | tt | zero_shot | aware-blind | 1w | mlp | 2 | 0.0014 | 0.0031 | 0 | 1 | mixed |
| aware-blind | tt | zero_shot | aware-blind | 1w | rf | 2 | -0.0002 | 0.0003 | 0 | 0 | mixed |
| aware-blind | tt | zero_shot | aware-blind | 1w | ridge | 2 | -0.0001 | 0.0000 | 2 | 0 | neg |
| aware-blind | tt | zero_shot | aware-blind | 1w | svr | 2 | 0.0003 | 0.0000 | 0 | 0 | pos |
| aware-blind | tt | zero_shot | aware-blind | 1w | xgb | 2 | 0.0008 | 0.0004 | 0 | 0 | pos |
| aware-blind | tt | zero_shot | aware-blind | bd | lasso | 2 | 0.0000 | 0.0000 | 0 | 0 | mixed |
| aware-blind | tt | zero_shot | aware-blind | bd | mlp | 2 | 0.0017 | 0.0003 | 0 | 0 | pos |
| aware-blind | tt | zero_shot | aware-blind | bd | rf | 2 | -0.0012 | 0.0001 | 1 | 0 | neg |
| aware-blind | tt | zero_shot | aware-blind | bd | ridge | 2 | -0.0001 | 0.0000 | 2 | 0 | neg |
| aware-blind | tt | zero_shot | aware-blind | bd | svr | 2 | 0.0008 | 0.0000 | 0 | 0 | pos |
| aware-blind | tt | zero_shot | aware-blind | bd | xgb | 2 | -0.0002 | 0.0003 | 1 | 0 | mixed |
| aware-blind | tt | zero_shot | aware-blind | none | absolute | 2 | 0.0001 | 0.0001 | 0 | 0 | pos |
| aware-blind | tt | zero_shot | aware-blind | none | delta | 2 | 0.0032 | 0.0007 | 0 | 1 | pos |
| bd-1w | mpi | full | aware | bd-1w | lasso | 2 | -0.0092 | 0.0085 | 1 | 0 | neg |
| bd-1w | mpi | full | aware | bd-1w | mlp | 2 | -0.0249 | 0.0228 | 1 | 0 | neg |
| bd-1w | mpi | full | aware | bd-1w | rf | 2 | 0.0006 | 0.0010 | 0 | 0 | mixed |
| bd-1w | mpi | full | aware | bd-1w | ridge | 2 | -0.0020 | 0.0034 | 0 | 0 | mixed |
| bd-1w | mpi | full | aware | bd-1w | svr | 2 | -0.0081 | 0.0231 | 0 | 0 | mixed |
| bd-1w | mpi | full | aware | bd-1w | xgb | 2 | -0.0078 | 0.0029 | 0 | 0 | neg |
| bd-1w | mpi | full | blind | bd-1w | lasso | 2 | -0.0091 | 0.0096 | 1 | 0 | neg |
| bd-1w | mpi | full | blind | bd-1w | mlp | 2 | -0.0189 | 0.0182 | 1 | 0 | neg |
| bd-1w | mpi | full | blind | bd-1w | rf | 2 | 0.0013 | 0.0004 | 0 | 0 | pos |
| bd-1w | mpi | full | blind | bd-1w | ridge | 2 | -0.0022 | 0.0032 | 0 | 0 | mixed |
| bd-1w | mpi | full | blind | bd-1w | svr | 2 | -0.0239 | 0.0109 | 0 | 0 | neg |
| bd-1w | mpi | full | blind | bd-1w | xgb | 2 | -0.0033 | 0.0072 | 1 | 0 | mixed |
| bd-1w | mpi | one_known | aware | bd-1w | lasso | 2 | -0.0011 | 0.0000 | 0 | 0 | neg |
| bd-1w | mpi | one_known | aware | bd-1w | mlp | 2 | 0.0333 | 0.0380 | 0 | 1 | pos |
| bd-1w | mpi | one_known | aware | bd-1w | rf | 2 | 0.0021 | 0.0007 | 0 | 0 | pos |
| bd-1w | mpi | one_known | aware | bd-1w | ridge | 2 | -0.0094 | 0.0000 | 0 | 0 | neg |
| bd-1w | mpi | one_known | aware | bd-1w | svr | 2 | 0.0053 | 0.0000 | 0 | 2 | pos |
| bd-1w | mpi | one_known | aware | bd-1w | xgb | 2 | -0.0032 | 0.0004 | 0 | 0 | neg |
| bd-1w | mpi | one_known | blind | bd-1w | lasso | 2 | -0.0006 | 0.0000 | 0 | 0 | neg |
| bd-1w | mpi | one_known | blind | bd-1w | mlp | 2 | -0.0086 | 0.0002 | 0 | 0 | neg |
| bd-1w | mpi | one_known | blind | bd-1w | rf | 2 | 0.0017 | 0.0003 | 0 | 0 | pos |
| bd-1w | mpi | one_known | blind | bd-1w | ridge | 2 | -0.0091 | 0.0000 | 0 | 0 | neg |
| bd-1w | mpi | one_known | blind | bd-1w | svr | 2 | 0.0101 | 0.0000 | 0 | 2 | pos |
| bd-1w | mpi | one_known | blind | bd-1w | xgb | 2 | -0.0011 | 0.0003 | 0 | 0 | neg |
| bd-1w | mpi | zero_shot | aware | bd-1w | lasso | 2 | 0.0037 | 0.0000 | 0 | 0 | pos |
| bd-1w | mpi | zero_shot | aware | bd-1w | mlp | 2 | -0.0040 | 0.0673 | 0 | 0 | mixed |
| bd-1w | mpi | zero_shot | aware | bd-1w | rf | 2 | 0.0184 | 0.0007 | 0 | 2 | pos |
| bd-1w | mpi | zero_shot | aware | bd-1w | ridge | 2 | -0.0205 | 0.0000 | 0 | 0 | neg |
| bd-1w | mpi | zero_shot | aware | bd-1w | svr | 2 | 0.0090 | 0.0000 | 0 | 0 | pos |
| bd-1w | mpi | zero_shot | aware | bd-1w | xgb | 2 | -0.0032 | 0.0001 | 0 | 0 | neg |
| bd-1w | mpi | zero_shot | blind | bd-1w | lasso | 2 | 0.0038 | 0.0000 | 0 | 0 | pos |
| bd-1w | mpi | zero_shot | blind | bd-1w | mlp | 2 | -0.0474 | 0.0756 | 1 | 0 | mixed |
| bd-1w | mpi | zero_shot | blind | bd-1w | rf | 2 | 0.0157 | 0.0034 | 0 | 2 | pos |
| bd-1w | mpi | zero_shot | blind | bd-1w | ridge | 2 | -0.0205 | 0.0000 | 0 | 0 | neg |
| bd-1w | mpi | zero_shot | blind | bd-1w | svr | 2 | 0.0116 | 0.0000 | 0 | 0 | pos |
| bd-1w | mpi | zero_shot | blind | bd-1w | xgb | 2 | 0.0012 | 0.0012 | 0 | 0 | pos |
| bd-1w | tt | full | aware | bd-1w | lasso | 2 | 0.0006 | 0.0019 | 0 | 0 | mixed |
| bd-1w | tt | full | aware | bd-1w | mlp | 2 | -0.0027 | 0.0036 | 0 | 0 | neg |
| bd-1w | tt | full | aware | bd-1w | rf | 2 | -0.0018 | 0.0011 | 0 | 0 | neg |
| bd-1w | tt | full | aware | bd-1w | ridge | 2 | 0.0021 | 0.0024 | 0 | 1 | pos |
| bd-1w | tt | full | aware | bd-1w | svr | 2 | -0.0014 | 0.0031 | 0 | 0 | mixed |
| bd-1w | tt | full | aware | bd-1w | xgb | 2 | 0.0016 | 0.0002 | 0 | 0 | pos |
| bd-1w | tt | full | blind | bd-1w | lasso | 2 | 0.0006 | 0.0019 | 0 | 0 | mixed |
| bd-1w | tt | full | blind | bd-1w | mlp | 2 | -0.0018 | 0.0056 | 0 | 0 | mixed |
| bd-1w | tt | full | blind | bd-1w | rf | 2 | -0.0002 | 0.0011 | 0 | 0 | mixed |
| bd-1w | tt | full | blind | bd-1w | ridge | 2 | 0.0021 | 0.0024 | 0 | 1 | pos |
| bd-1w | tt | full | blind | bd-1w | svr | 2 | 0.0002 | 0.0011 | 0 | 0 | mixed |
| bd-1w | tt | full | blind | bd-1w | xgb | 2 | -0.0002 | 0.0006 | 0 | 0 | mixed |
| bd-1w | tt | one_known | aware | bd-1w | lasso | 2 | 0.0011 | 0.0000 | 0 | 2 | pos |
| bd-1w | tt | one_known | aware | bd-1w | mlp | 2 | 0.0019 | 0.0009 | 0 | 0 | pos |
| bd-1w | tt | one_known | aware | bd-1w | rf | 2 | 0.0012 | 0.0000 | 0 | 2 | pos |
| bd-1w | tt | one_known | aware | bd-1w | ridge | 2 | 0.0035 | 0.0000 | 0 | 2 | pos |
| bd-1w | tt | one_known | aware | bd-1w | svr | 2 | 0.0009 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | one_known | aware | bd-1w | xgb | 2 | -0.0000 | 0.0006 | 0 | 0 | mixed |
| bd-1w | tt | one_known | blind | bd-1w | lasso | 2 | 0.0011 | 0.0000 | 0 | 2 | pos |
| bd-1w | tt | one_known | blind | bd-1w | mlp | 2 | 0.0016 | 0.0032 | 0 | 0 | mixed |
| bd-1w | tt | one_known | blind | bd-1w | rf | 2 | 0.0007 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | one_known | blind | bd-1w | ridge | 2 | 0.0036 | 0.0000 | 0 | 2 | pos |
| bd-1w | tt | one_known | blind | bd-1w | svr | 2 | 0.0004 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | one_known | blind | bd-1w | xgb | 2 | 0.0001 | 0.0005 | 0 | 0 | mixed |
| bd-1w | tt | zero_shot | aware | bd-1w | lasso | 2 | 0.0019 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | aware | bd-1w | mlp | 2 | -0.0009 | 0.0033 | 0 | 0 | mixed |
| bd-1w | tt | zero_shot | aware | bd-1w | rf | 2 | 0.0017 | 0.0006 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | aware | bd-1w | ridge | 2 | 0.0005 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | aware | bd-1w | svr | 2 | 0.0020 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | aware | bd-1w | xgb | 2 | 0.0036 | 0.0001 | 0 | 2 | pos |
| bd-1w | tt | zero_shot | blind | bd-1w | lasso | 2 | 0.0019 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | blind | bd-1w | mlp | 2 | -0.0013 | 0.0004 | 0 | 0 | neg |
| bd-1w | tt | zero_shot | blind | bd-1w | rf | 2 | 0.0026 | 0.0004 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | blind | bd-1w | ridge | 2 | 0.0004 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | blind | bd-1w | svr | 2 | 0.0015 | 0.0000 | 0 | 0 | pos |
| bd-1w | tt | zero_shot | blind | bd-1w | xgb | 2 | 0.0046 | 0.0008 | 0 | 2 | pos |
| method-baseline | mpi | full | aware | 1w | lasso | 2 | -0.0411 | 0.0060 | 0 | 0 | neg |
| method-baseline | mpi | full | aware | 1w | mlp | 2 | -0.0728 | 0.0088 | 0 | 0 | neg |
| method-baseline | mpi | full | aware | 1w | rf | 2 | -0.0881 | 0.0078 | 2 | 0 | neg |
| method-baseline | mpi | full | aware | 1w | ridge | 2 | -0.0456 | 0.0022 | 0 | 0 | neg |
| method-baseline | mpi | full | aware | 1w | svr | 2 | -0.0724 | 0.0160 | 1 | 0 | neg |
| method-baseline | mpi | full | aware | 1w | xgb | 2 | -0.0922 | 0.0098 | 2 | 0 | neg |
| method-baseline | mpi | full | aware | bd | lasso | 2 | -0.0503 | 0.0026 | 0 | 0 | neg |
| method-baseline | mpi | full | aware | bd | mlp | 2 | -0.0977 | 0.0140 | 1 | 0 | neg |
| method-baseline | mpi | full | aware | bd | rf | 2 | -0.0876 | 0.0089 | 2 | 0 | neg |
| method-baseline | mpi | full | aware | bd | ridge | 2 | -0.0476 | 0.0011 | 0 | 0 | neg |
| method-baseline | mpi | full | aware | bd | svr | 2 | -0.0805 | 0.0070 | 1 | 0 | neg |
| method-baseline | mpi | full | aware | bd | xgb | 2 | -0.1000 | 0.0070 | 1 | 0 | neg |
| method-baseline | mpi | full | aware | none | absolute | 2 | -0.0343 | 0.0421 | 0 | 0 | neg |
| method-baseline | mpi | full | aware | none | delta | 2 | -0.0458 | 0.0154 | 0 | 0 | neg |
| method-baseline | mpi | full | blind | 1w | lasso | 2 | -0.0411 | 0.0065 | 0 | 0 | neg |
| method-baseline | mpi | full | blind | 1w | mlp | 2 | -0.0767 | 0.0020 | 0 | 0 | neg |
| method-baseline | mpi | full | blind | 1w | rf | 2 | -0.0847 | 0.0126 | 1 | 0 | neg |
| method-baseline | mpi | full | blind | 1w | ridge | 2 | -0.0456 | 0.0020 | 0 | 0 | neg |
| method-baseline | mpi | full | blind | 1w | svr | 2 | -0.0560 | 0.0011 | 1 | 0 | neg |
| method-baseline | mpi | full | blind | 1w | xgb | 2 | -0.0786 | 0.0179 | 1 | 0 | neg |
| method-baseline | mpi | full | blind | bd | lasso | 2 | -0.0502 | 0.0032 | 0 | 0 | neg |
| method-baseline | mpi | full | blind | bd | mlp | 2 | -0.0956 | 0.0162 | 1 | 0 | neg |
| method-baseline | mpi | full | blind | bd | rf | 2 | -0.0834 | 0.0122 | 1 | 0 | neg |
| method-baseline | mpi | full | blind | bd | ridge | 2 | -0.0478 | 0.0012 | 0 | 0 | neg |
| method-baseline | mpi | full | blind | bd | svr | 2 | -0.0799 | 0.0098 | 1 | 0 | neg |
| method-baseline | mpi | full | blind | bd | xgb | 2 | -0.0820 | 0.0106 | 2 | 0 | neg |
| method-baseline | mpi | full | blind | none | absolute | 2 | -0.0417 | 0.0357 | 0 | 0 | neg |
| method-baseline | mpi | full | blind | none | delta | 2 | -0.0738 | 0.0290 | 0 | 0 | neg |
| method-baseline | mpi | one_known | aware | 1w | lasso | 2 | -0.0254 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | 1w | mlp | 2 | 0.0013 | 0.0107 | 0 | 0 | mixed |
| method-baseline | mpi | one_known | aware | 1w | rf | 2 | -0.0361 | 0.0003 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | 1w | ridge | 2 | -0.0256 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | 1w | svr | 2 | -0.0223 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | 1w | xgb | 2 | -0.0332 | 0.0008 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | bd | lasso | 2 | -0.0265 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | bd | mlp | 2 | 0.0346 | 0.0273 | 0 | 0 | pos |
| method-baseline | mpi | one_known | aware | bd | rf | 2 | -0.0340 | 0.0004 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | bd | ridge | 2 | -0.0350 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | bd | svr | 2 | -0.0170 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | bd | xgb | 2 | -0.0364 | 0.0011 | 2 | 0 | neg |
| method-baseline | mpi | one_known | aware | none | absolute | 2 | 0.0434 | 0.0037 | 0 | 1 | pos |
| method-baseline | mpi | one_known | aware | none | delta | 2 | -0.0293 | 0.0043 | 0 | 0 | neg |
| method-baseline | mpi | one_known | blind | 1w | lasso | 2 | -0.0247 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | 1w | mlp | 2 | -0.0054 | 0.0111 | 0 | 0 | mixed |
| method-baseline | mpi | one_known | blind | 1w | rf | 2 | -0.0321 | 0.0004 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | 1w | ridge | 2 | -0.0253 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | 1w | svr | 2 | -0.0222 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | 1w | xgb | 2 | -0.0310 | 0.0009 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | bd | lasso | 2 | -0.0253 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | bd | mlp | 2 | -0.0140 | 0.0112 | 0 | 0 | neg |
| method-baseline | mpi | one_known | blind | bd | rf | 2 | -0.0304 | 0.0001 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | bd | ridge | 2 | -0.0345 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | bd | svr | 2 | -0.0121 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | bd | xgb | 2 | -0.0321 | 0.0006 | 2 | 0 | neg |
| method-baseline | mpi | one_known | blind | none | absolute | 2 | 0.0371 | 0.0084 | 0 | 1 | pos |
| method-baseline | mpi | one_known | blind | none | delta | 2 | -0.0336 | 0.0018 | 0 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | 1w | lasso | 2 | -0.0345 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | 1w | mlp | 2 | -0.0017 | 0.0155 | 0 | 0 | mixed |
| method-baseline | mpi | zero_shot | aware | 1w | rf | 2 | -0.0578 | 0.0012 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | 1w | ridge | 2 | -0.0321 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | 1w | svr | 2 | -0.0246 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | 1w | xgb | 2 | -0.0488 | 0.0005 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | bd | lasso | 2 | -0.0307 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | bd | mlp | 2 | -0.0057 | 0.0518 | 0 | 0 | mixed |
| method-baseline | mpi | zero_shot | aware | bd | rf | 2 | -0.0394 | 0.0006 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | bd | ridge | 2 | -0.0527 | 0.0000 | 0 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | bd | svr | 2 | -0.0156 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | bd | xgb | 2 | -0.0519 | 0.0004 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | aware | none | absolute | 2 | 0.1485 | 0.0245 | 0 | 1 | pos |
| method-baseline | mpi | zero_shot | aware | none | delta | 2 | -0.0785 | 0.0095 | 0 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | 1w | lasso | 2 | -0.0331 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | 1w | mlp | 2 | -0.0195 | 0.0473 | 0 | 0 | mixed |
| method-baseline | mpi | zero_shot | blind | 1w | rf | 2 | -0.0513 | 0.0001 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | 1w | ridge | 2 | -0.0318 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | 1w | svr | 2 | -0.0229 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | 1w | xgb | 2 | -0.0488 | 0.0016 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | bd | lasso | 2 | -0.0294 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | bd | mlp | 2 | -0.0670 | 0.0283 | 0 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | bd | rf | 2 | -0.0356 | 0.0035 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | bd | ridge | 2 | -0.0523 | 0.0000 | 0 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | bd | svr | 2 | -0.0113 | 0.0000 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | bd | xgb | 2 | -0.0476 | 0.0004 | 2 | 0 | neg |
| method-baseline | mpi | zero_shot | blind | none | absolute | 2 | 0.3667 | 0.2318 | 0 | 2 | pos |
| method-baseline | mpi | zero_shot | blind | none | delta | 2 | -0.0561 | 0.0381 | 0 | 0 | neg |
| method-baseline | tt | full | aware | 1w | lasso | 2 | -0.0093 | 0.0014 | 0 | 0 | neg |
| method-baseline | tt | full | aware | 1w | mlp | 2 | -0.0152 | 0.0013 | 0 | 0 | neg |
| method-baseline | tt | full | aware | 1w | rf | 2 | -0.0141 | 0.0010 | 2 | 0 | neg |
| method-baseline | tt | full | aware | 1w | ridge | 2 | -0.0104 | 0.0006 | 1 | 0 | neg |
| method-baseline | tt | full | aware | 1w | svr | 2 | -0.0145 | 0.0008 | 1 | 0 | neg |
| method-baseline | tt | full | aware | 1w | xgb | 2 | -0.0157 | 0.0016 | 2 | 0 | neg |
| method-baseline | tt | full | aware | bd | lasso | 2 | -0.0087 | 0.0006 | 1 | 0 | neg |
| method-baseline | tt | full | aware | bd | mlp | 2 | -0.0179 | 0.0023 | 1 | 0 | neg |
| method-baseline | tt | full | aware | bd | rf | 2 | -0.0159 | 0.0001 | 2 | 0 | neg |
| method-baseline | tt | full | aware | bd | ridge | 2 | -0.0083 | 0.0018 | 0 | 0 | neg |
| method-baseline | tt | full | aware | bd | svr | 2 | -0.0159 | 0.0023 | 1 | 0 | neg |
| method-baseline | tt | full | aware | bd | xgb | 2 | -0.0141 | 0.0014 | 2 | 0 | neg |
| method-baseline | tt | full | aware | none | absolute | 2 | 0.0016 | 0.0017 | 0 | 0 | pos |
| method-baseline | tt | full | aware | none | delta | 2 | -0.0151 | 0.0008 | 0 | 0 | neg |
| method-baseline | tt | full | blind | 1w | lasso | 2 | -0.0093 | 0.0014 | 0 | 0 | neg |
| method-baseline | tt | full | blind | 1w | mlp | 2 | -0.0151 | 0.0020 | 0 | 0 | neg |
| method-baseline | tt | full | blind | 1w | rf | 2 | -0.0140 | 0.0005 | 2 | 0 | neg |
| method-baseline | tt | full | blind | 1w | ridge | 2 | -0.0104 | 0.0007 | 1 | 0 | neg |
| method-baseline | tt | full | blind | 1w | svr | 2 | -0.0151 | 0.0013 | 1 | 0 | neg |
| method-baseline | tt | full | blind | 1w | xgb | 2 | -0.0154 | 0.0019 | 2 | 0 | neg |
| method-baseline | tt | full | blind | bd | lasso | 2 | -0.0087 | 0.0006 | 1 | 0 | neg |
| method-baseline | tt | full | blind | bd | mlp | 2 | -0.0169 | 0.0036 | 1 | 0 | neg |
| method-baseline | tt | full | blind | bd | rf | 2 | -0.0142 | 0.0005 | 2 | 0 | neg |
| method-baseline | tt | full | blind | bd | ridge | 2 | -0.0083 | 0.0017 | 0 | 0 | neg |
| method-baseline | tt | full | blind | bd | svr | 2 | -0.0149 | 0.0002 | 1 | 0 | neg |
| method-baseline | tt | full | blind | bd | xgb | 2 | -0.0157 | 0.0025 | 2 | 0 | neg |
| method-baseline | tt | full | blind | none | absolute | 2 | 0.0032 | 0.0033 | 0 | 0 | pos |
| method-baseline | tt | full | blind | none | delta | 2 | -0.0134 | 0.0063 | 1 | 0 | neg |
| method-baseline | tt | one_known | aware | 1w | lasso | 2 | -0.0027 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | 1w | mlp | 2 | -0.0041 | 0.0004 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | 1w | rf | 2 | -0.0061 | 0.0001 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | 1w | ridge | 2 | -0.0046 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | 1w | svr | 2 | -0.0034 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | 1w | xgb | 2 | -0.0061 | 0.0006 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | bd | lasso | 2 | -0.0016 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | one_known | aware | bd | mlp | 2 | -0.0022 | 0.0013 | 0 | 0 | neg |
| method-baseline | tt | one_known | aware | bd | rf | 2 | -0.0049 | 0.0001 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | bd | ridge | 2 | -0.0010 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | one_known | aware | bd | svr | 2 | -0.0026 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | bd | xgb | 2 | -0.0061 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | aware | none | absolute | 2 | 0.0026 | 0.0016 | 0 | 0 | pos |
| method-baseline | tt | one_known | aware | none | delta | 2 | 0.0002 | 0.0018 | 0 | 0 | mixed |
| method-baseline | tt | one_known | blind | 1w | lasso | 2 | -0.0027 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | 1w | mlp | 2 | -0.0042 | 0.0014 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | 1w | rf | 2 | -0.0058 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | 1w | ridge | 2 | -0.0045 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | 1w | svr | 2 | -0.0033 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | 1w | xgb | 2 | -0.0062 | 0.0008 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | bd | lasso | 2 | -0.0016 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | one_known | blind | bd | mlp | 2 | -0.0026 | 0.0046 | 1 | 0 | mixed |
| method-baseline | tt | one_known | blind | bd | rf | 2 | -0.0051 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | bd | ridge | 2 | -0.0009 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | one_known | blind | bd | svr | 2 | -0.0030 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | bd | xgb | 2 | -0.0061 | 0.0003 | 2 | 0 | neg |
| method-baseline | tt | one_known | blind | none | absolute | 2 | 0.0032 | 0.0052 | 0 | 0 | mixed |
| method-baseline | tt | one_known | blind | none | delta | 2 | -0.0040 | 0.0009 | 0 | 0 | neg |
| method-baseline | tt | zero_shot | aware | 1w | lasso | 2 | -0.0026 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | 1w | mlp | 2 | -0.0068 | 0.0007 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | 1w | rf | 2 | -0.0068 | 0.0005 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | 1w | ridge | 2 | -0.0052 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | 1w | svr | 2 | -0.0041 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | 1w | xgb | 2 | -0.0086 | 0.0001 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | bd | lasso | 2 | -0.0007 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | zero_shot | aware | bd | mlp | 2 | -0.0077 | 0.0026 | 1 | 0 | neg |
| method-baseline | tt | zero_shot | aware | bd | rf | 2 | -0.0051 | 0.0001 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | bd | ridge | 2 | -0.0047 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | zero_shot | aware | bd | svr | 2 | -0.0022 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | aware | bd | xgb | 2 | -0.0049 | 0.0002 | 1 | 0 | neg |
| method-baseline | tt | zero_shot | aware | none | absolute | 2 | -0.0026 | 0.0001 | 0 | 0 | neg |
| method-baseline | tt | zero_shot | aware | none | delta | 2 | -0.0252 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | 1w | lasso | 2 | -0.0026 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | 1w | mlp | 2 | -0.0082 | 0.0024 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | 1w | rf | 2 | -0.0066 | 0.0002 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | 1w | ridge | 2 | -0.0051 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | 1w | svr | 2 | -0.0044 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | 1w | xgb | 2 | -0.0094 | 0.0003 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | bd | lasso | 2 | -0.0007 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | zero_shot | blind | bd | mlp | 2 | -0.0095 | 0.0028 | 1 | 0 | neg |
| method-baseline | tt | zero_shot | blind | bd | rf | 2 | -0.0039 | 0.0002 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | bd | ridge | 2 | -0.0046 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | zero_shot | blind | bd | svr | 2 | -0.0029 | 0.0000 | 2 | 0 | neg |
| method-baseline | tt | zero_shot | blind | bd | xgb | 2 | -0.0048 | 0.0005 | 1 | 0 | neg |
| method-baseline | tt | zero_shot | blind | none | absolute | 2 | -0.0027 | 0.0000 | 0 | 0 | neg |
| method-baseline | tt | zero_shot | blind | none | delta | 2 | -0.0283 | 0.0007 | 2 | 0 | neg |

## HP grid edges (share of tasks choosing the lowest / highest value; flag > 30%)

| method | axis | values | n_tasks | frac_low_edge | frac_high_edge | categorical_shares | flag_gt30 |
|---|---|---|---|---|---|---|---|
| absolute | lr | [0.001, 0.003] | 16 | 0.12 | 0.88 | nan | True |
| absolute | weight_decay | [0.0001, 0.01] | 16 | 0.12 | 0.88 | nan | True |
| absolute | epochs | [10.0, 20.0, 40.0, 80.0, 160.0] | 16 | 0.00 | 0.12 | nan | False |
| delta | lr | [0.001, 0.003] | 16 | 0.25 | 0.75 | nan | True |
| delta | weight_decay | [0.0001, 0.01] | 16 | 0.44 | 0.56 | nan | True |
| delta | epochs | [10.0, 20.0, 40.0, 80.0, 160.0] | 16 | 0.06 | 0.31 | nan | True |
| delta | agg | ['kernel_0.3', 'kernel_1', 'kernel_3', 'median', 'uniform'] | 16 | nan | nan | {'kernel_1': 0.625, 'kernel_3': 0.312, 'median': 0.062} | False |
| lasso | alpha | [0.001, 0.003, 0.01, 0.03, 0.1, 0.3] | 32 | 0.00 | 0.31 | nan | True |
| mlp | weight_decay | [0.0001, 0.01] | 32 | 0.34 | 0.66 | nan | True |
| mlp | epochs | [100.0, 300.0, 1000.0] | 32 | 0.72 | 0.28 | nan | True |
| mlp | hidden | ['16', '32-16'] | 32 | nan | nan | {'32-16': 0.844, '16': 0.156} | False |
| rf | max_depth | [3.0, 6.0] | 32 | 0.38 | 0.19 | nan | True |
| rf | min_samples_leaf | [1.0, 3.0, 5.0] | 32 | 0.56 | 0.12 | nan | True |
| rf | max_features | [0.33, 1.0] | 32 | 0.34 | 0.66 | nan | True |
| ridge | alpha | [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] | 32 | 0.00 | 0.31 | nan | True |
| svr | C | [0.1, 1.0, 10.0] | 32 | 0.16 | 0.41 | nan | True |
| svr | epsilon | [0.05, 0.2] | 32 | 0.50 | 0.50 | nan | True |
| svr | gamma | ['0.3/p', 'scale'] | 32 | nan | nan | {'0.3/p': 0.688, 'scale': 0.312} | False |
| xgb | max_depth | [2.0, 4.0] | 32 | 0.69 | 0.31 | nan | True |
| xgb | n_estimators | [100.0, 300.0] | 32 | 0.81 | 0.19 | nan | True |
| xgb | learning_rate | [0.03, 0.1] | 32 | 0.78 | 0.22 | nan | True |
| xgb | min_child_weight | [1.0, 5.0] | 32 | 0.50 | 0.50 | nan | True |
