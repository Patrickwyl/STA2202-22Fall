```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(astsa)
set.seed(17)
```

# 1. Fit ARIMA model to $S_t$, the monthly sales data.

## 1. Plot the series and check potential transformation and differences

```{r}
S_t <- ts(sales, frequency = 12, start = 1970)  # monthly records
par(mfrow=c(3,1))
plot.ts(S_t, main="monthly sales", ylab="sales")
plot.ts(log(S_t), main="log monthly sales", ylab="log(sales)")
plot.ts(diff(log(S_t)), main="First difference of log monthly sales", ylab="diff(log((sales))")
abline(h=mean(diff(log(S_t))), lwd=2, col=2)  # mean line
```

Since the series does NOT tend to be stationary after taking log transformation and then the first difference, so we do NOT consider taking log transformations.

```{r}
par(mfrow=c(2,1))
plot.ts(diff(S_t), main="First difference of monthly sales", ylab="diff(sales)")
abline(h=mean(diff(S_t)), lwd=2, col=2)  # mean line
plot.ts(diff(diff(S_t)), main="Second difference of monthly sales", ylab="diff(sales, 2)")
abline(h=mean(diff(diff(S_t))), lwd=2, col=2)  # mean line
```

Since the series appears to be stationary with a constant variance level after taking the second difference, so we are now ready to fit a model.

## 2. Identify possible models
```{r, results=FALSE}
acf2(diff(diff(S_t)), main="Sample ACF and PACF of the second difference of monthly sales")
```

Inspecting the sample ACF and PACF, indicate that the sample ACF is cutting off after lag h=1, whereas the sample PACF is damped exponential dying down with oscillation. This would suggest the second difference follows an MA(1) process, q=1. 

Inspecting the sample ACF and PACF, indicate that the sample PACF is cutting off after lag h=3, whereas the sample ACF is damped exponential dying down with oscillation. This would suggest the second difference follows an AR(3) process, p=3. 

However, both sample ACF and PACF have an extreme spike at lag 1 and appear as though both are tailing off, we decide to fit both an ARIMA(1,2,1) and an ARIMA(3,2,1) for the monthly sales series.

## 3. Fit model and estimate model parameters
```{r}
arima_1 <- arima(S_t, order=c(1,2,1))
arima_1
arima_2 <- arima(S_t, order=c(3,2,1))
arima_2
```

## 4. Perform diagnostic checks and model choice
```{r}
tsdiag(arima_1)
tsdiag(arima_2)
```

1. The time plots of the standardized residuals for both fitted models show no obvious patterns. \
2. The ACF of the standardized residuals for both fitted models show no apparent departure from the model assumptions.\
3. The portmanteau test statistics for both fitted models are never significant at the lags shown, which means the residuals are coming from a White Noise series.

```{r}
par(mfrow=c(1,2))
qqnorm(arima_1$resid); qqline(arima_1$resid)
qqnorm(arima_2$resid); qqline(arima_2$resid)
```

Both normal Q-Q plots show that residuals are approximately normally distributed as most points lie close to the diagonal line with very few random fluctuations at extreme ends.

We can run the Shapiro-Wilk normality test to do a further diagnostic.

```{r}
shapiro.test(arima_1$resid)
shapiro.test(arima_2$resid)
```

Since the p-values of both Shapiro-Wilk normality tests are greater than 0.05, so we do NOT reject the null hypothesis, then we conclude that the residuals are normally distributed.

Hence, both models appear to fit the monthly sales well. To choose a final model, we compare the AIC for both models.

```{r}
arima_1$aic
arima_2$aic
```

Thus, we choose ARIMA(1,2,1) as our final model as it has a lower AIC.

# 2. Use the cross-correlation function (CCF) and lag plots between $\nabla S_t$ and $\nabla L_t$
```{r}
L_t <- ts(lead, frequency = 12, start = 1970)  # monthly records
lag2.plot(diff(L_t), diff(S_t), max.lag = 8)
```

The above scatter plots show the $\nabla S_t$ against the lagged $\nabla L_t$ at lag up to 8 months, with sample cross-correlations in the upper right corner and the Lowess fits in red. We can see that they indicate a strong positive linear relationship between $\nabla S_t$ and $\nabla L_t$ at lag $h=3$ with a sample cross-correlation of 0.72. Thus, we can consider fitting a regression model of $\nabla S_t$ and $\nabla L_{t-3}$.

# 3. Fit the regression model $\nabla S_t = \beta_0 + \beta_1 \nabla L_{t-3} +x_t$, where $x_t$ is an ARMA process

## 1. First, run an ordinary regression of $\nabla S_t$ on $\nabla L_{t-3}$.
```{r}
# combine data frames
df <- ts.intersect(x=lag(L_t, -3), y=diff(S_t), dframe=T)
ordinary_model <- lm(y ~ x, data=df, na.action=NULL)  # lm with intercept
```

## 2. Retain the residuals, $\hat x_t = \nabla S_t - \hat \beta_0 - \hat \beta_1 \nabla L_{t-3}$.
```{r}
par(mfrow=c(2,1))
plot(resid(ordinary_model), main="Plot of fitted model residuals", ylab="residual")
diff_resid <- diff(resid(ordinary_model))
plot(diff_resid, main="First difference of fitted model residuals", ylab="diff(residual)")
abline(h=mean(diff_resid), lwd=2, col=2)
```

The residual series looks stationary after taking the first difference. Then we can look into the sample ACF and sample PACF of the first difference.

## 3. Identify ARMA model for the residual $\hat x_t$.

```{r, results=FALSE}
acf2(diff_resid, main="samle ACF of the first difference of residuals")
```

Inspecting the sample ACF and PACF, indicate that the sample ACF is cutting off after lag h=1, whereas the sample PACF is damped exponential dying down with oscillation. This would suggest the residual follows an MA(1) process, q=1. 

Inspecting the sample ACF and PACF, indicate that the sample PACF is cutting off after lag h=3, whereas the sample ACF is damped exponential dying down with oscillation. This would suggest the residual follows an AR(3) process, p=3. 

However, both sample ACF and PACF have an extreme spike at lag 1 and it appears as though both are tailing off, so we decide to fit both an ARIMA(1,1,1) and an ARIMA(3,1,1) for the residual series.

```{r, warning=FALSE}
resid_arima_1 <- arima(diff_resid, order=c(1,1,1))
resid_arima_1
resid_arima_2 <- arima(diff_resid, order=c(3,1,1))
resid_arima_2
```

Then we choose the ARIMA(3,1,1) as an appropriate model for residual series as it has a lower AIC.

## 4. Run weighted least squares (or MLE) on the regression model with autocorrelated errors.

```{r}
final_model <- arima(df$y, order=c(3,1,1), xreg=df$x)
final_model
```

We can compute the 95% confidence intervals for estimated coefficients.
```{r}
# standard errors of coefficients
se <- sqrt(diag(final_model$var.coef))
# 95% CI of estimated coefficients
coef_CI <- cbind(final_model$coef -1.96*se, final_model$coef + 1.96*se)
colnames(coef_CI) <- c("2.5%", "97.5%")
coef_CI
```

Since all 95% confidence intervals do NOT contain 0 except the variable "ar3", so this indicates that our final regression model has four statistically significant independent variables and one insignificant. Thus, this model can be further improved.