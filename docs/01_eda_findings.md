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

# Model comparison

I compared 5 models on 4 sets of features. I used 5-fold cross-validation repeated 10 times, so every model got 50 scores. My baseline was a simple linear model that only uses R&D spend. It got an RMSE of about 9,369 and an R² of 0.928.

On paper the best one was Extra Trees on the three raw spends, with an RMSE of 9,007 and an R² of 0.936. That is only about 360 better than the baseline. The baseline's own spread is about 2,750, so a gap that small could easily be luck. No model beat the baseline by more than that spread, and the tree models were not better than the linear ones.

Adding admin and marketing to the linear model did not help either (9,432 compared with 9,369). The features built from shares were worse, because they leave out how much was spent on R&D. The linear model on shares only got an R² of 0.70.

## Without the four zero rows
When I took out the four startups with a zero spend, the errors for the linear models went down: from 9,369 to 7,818 for R&D only, and from 9,432 to 7,602 for all three spends. The spread got smaller too. But the order of the models stayed the same, and the linear models were still as good as or better than Extra Trees. So those four rows make predictions harder, but they don't change my main conclusion.

## What the coefficients say
In the linear model with all three spends:
- R&D: 0.81 (range 0.70 to 0.89). One more dollar on R&D goes with about 81 cents more profit.
- Marketing: 0.03 (range 0.002 to 0.06). A very small effect, and the range only just stays above zero, so I am not sure about it.
- Admin: -0.03 (range -0.13 to 0.06). The range includes zero, so I can't say admin matters.

This is only a link in 50 startups. It does not prove that spending more on R&D makes profit go up.

## Where the model is wrong
The predictions follow the red line fairly well, with more scatter at the low-profit end, and the errors look random. The biggest mistake is row 49, off by about 35,500. It spent nothing on R&D and made only 14,681, but the model predicts around 50,000 for a startup with no R&D. The next biggest errors are about 16,000 to 19,000 (rows 14, 45, 36 and 15), and none of them are zero-spend rows, so I don't know why they are off.

## What I will pick
A plain linear model is as accurate as the tree models and I can explain it, so I will use linear regression. I still need to decide whether to use R&D alone or all three spends.

# Final model

I picked a plain linear regression on all three spends. In my tests it was as accurate as the tree models, and I can explain it.

I trained it on all 50 startups and saved it in models/startup_profit_model.joblib. The typical error from cross-validation is about 9,764 (RMSE) and 7,175 (MAE). The coefficients are the same as before: about 0.81 for R&D, -0.03 for admin and 0.03 for marketing, with an intercept of about 50,122.

To show how unsure a prediction is, I built a 90% range from the errors in cross-validation. For one prediction the range goes from about 17,300 below to about 13,300 above. It is not the same on both sides, because the model's bigger mistakes are over-predictions (like row 49). I checked the range by building it without each startup and then testing on that startup. It covered about 89% of the cases, which is close to the 90% I wanted.

The range is the same width for every input, so it does not get wider for strange inputs. When someone enters a spend outside what the model has seen (R&D above 165,349, for example), the app will show a warning. At an R&D of 500,000 the model says about 455,000, and I would not trust that.