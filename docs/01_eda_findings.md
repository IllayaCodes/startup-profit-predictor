# What I found in the EDA

My data has 50 startups and 4 columns. There are no missing values and no duplicate rows, so the data is clean.

## The zeros
Four startups have a zero in some spend column (rows 19, 47, 48 and 49). Two have zero R&D and three have zero marketing. I decided to keep all 50 rows and treat the zeros as real values. One reason is row 19: it spent nothing on marketing but still made a profit of about 122,777, so a zero is not always bad data. Those four startups made 53,923 profit on average and the other 46 made 117,064. That points the same way as I expected, but four rows is too few to be sure.

## What matters for profit
R&D matters the most. Its correlation with profit is 0.99, and the range I got from bootstrapping is 0.97 to 0.995. Marketing is next at 0.72 (range 0.51 to 0.87). The two ranges do not overlap, so I can say R&D is stronger.

Admin is weak. Its correlation is 0.17 and the range goes from -0.12 to 0.43. Because that range includes 0, I can't say admin spending is linked to profit at all.

R&D and marketing are related to each other (0.72), but the VIF values are only about 2.4, so it is not a big problem. I will keep both.

## How the best startups spend
The top 25% by profit spend more money overall (about 578,000 compared with about 346,000 for the rest). They also split it differently: 55% on marketing, 23% on R&D and 22% on admin. The rest spend 44% on marketing, 14% on R&D and 42% on admin.

I need to be careful with this. The top group is only around 12 startups. Also, admin spend barely changes between startups, so its share goes down on its own when total spend goes up. That doesn't mean admin spending hurts profit.

## Other things I noticed
- No outliers in the spend columns. There is one in profit, row 49 (14,681, with zero R&D).
- Nothing looks curved in the scatter plots, so straight-line models are a fair start.

## What I will do in Step 3
- Keep the three spend columns.
- Add total spend and the share of each spend, using only the spend columns and never profit.
- Check my results again without the four zero rows.