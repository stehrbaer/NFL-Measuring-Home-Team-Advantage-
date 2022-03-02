# Predicting-NFL-Wins-Historical-
In this project, I used a variety of machine learning tools to predicting NFL Games, historically, using an NFL Kaggle Dataset. My goal of the project was to see if I could use spread-betting statistics to evaluate the impact of home-field advantage in the NFL. Please find below the descriptive process for the project. 

In every sport, playing in front of your own fans is considered an advantage. Home field advantage is a term that has been coined now for many generations throughout all team sports. But does playing in your own stadium, in front of crowds of fans cheering you on really make a difference? Throughout this report, we will be exploring the impact of home-field advantage in the National Football League (NFL) by exploring the impact on home-field advantage on scoring and gambling statistics. 

The National Football League (NFL) is the one of the most viewed sports leagues globally and generates billions of dollars in revenue annually. The legalization of online sports-betting throughout the US offers large benefits for the NFL, as the legalization is projected to not only increase the amount gambled on the league, but also lead to a projected 1.75 billion USD in new revenue through increased media-rights, sponsorship deals, increased fan-participation, and advertising1. 

The most popular method to bet on the NFL is spread betting, which is a method devised by mathematician Charles McNeil in 1940s. Spread betting aims to ‘even the odds’ between matchups, with the aim of creating an even-split of money bet between both teams2. In many cases, not just in the NFL, a team with superior talent will be favored over an ‘underdog’. The betting returns of picking the favorite will be limited, therefore, to increase the returns for the favored team and to create equal number of bettors between both teams, the bookmaker will assign a ‘spread’ that the favor-team must exceed, which will offer more favorable returns for the bettor, usually resulting in even to near-even returns for betting either side of the point-spread3. 

For example, heading into Week 15 of the 2007 NFL Season. The New England Patriots had amassed an undefeated record up until this point, going 14-0. They faced the New York Jets, who to this point had managed a record of 3-10. The Vegas bookmakers had installed the Patriots as clear favorites for this matchup, with the Patriots playing on their home-field, Gilette Stadium, with a far superior roster in terms of talent at their disposal. The spread for the Patriots was set at –20.5, which meant the Patriots would have to score 21 points more than the Jets to ‘cover’ the spread. On the opposite spectrum, the New York Jets have a 20.5 point ‘handicap’. The game ended with the score of 20-10 for the New England Patriots, with the Patriots extending their unbeaten streak but failing to ‘cover’ the spread4.  

 Data Collection and Cleaning 

All data was collected via Kaggle5, consisting of an 18 x 13187 table initially. After cleaning the dataset, we were left with 7688 observations per column. Initial relevant variables available being; 

Schedule-date – the date of the matchup 

Schedule-season – the season for the matchup  

Schedule-week – NFL Season consists of 17 weeks; weeks go from 1-17 

Schedule-playoff – whether the matchup was taking place in the regular season or playoffs(T/F) 

Team-home – the name of home team in the matchup 

Score-home – the score of the home team 

Team-away – the name of the away team 

Score-away – score of the away team 

Team-favorite-id – the short-form of the spread favorite team 

Spread-favorite – the value of spread for the favored team 

Over-under-line – the projected total score for the matchup  

Additionally, several variables were added to further the analysis, these consisted of the absolute value of the spread, as spread for favorited teams is expressed with a negative value to signify the additional points the team must exceed in order to ‘cover’ the spread, variable ‘actual-result’ which represents the actual score difference, score-home and score-away, from the home-team's perspective, and several binary classifiers such as: 

Home-team-fav – if the team-id for favored team matches the home-team name 

Covered-spread – if the favored team covered the spread (variable actual-result must be greater than spread in absolute terms)  

Following the addition of other variables, the dataset was split into two, one consisting of matchups covering the NFL Regular Season, and the other consisting of NFL Playoff only matchups. The purpose behind this is further capture the impact of home-team advantage. As mentioned before, the NFL Regular Season consists of 17 weeks, where 32-teams, which are split into two conferences, and then further divided into 4 divisions with 4 teams per division, face-off in pre-determined matchups to determine playoff qualification. To qualify, a team must finish with the best record in their respective division, where each division winner is seeded 1-4 based on number of wins or can qualify for one of 6 wild-card slots by being the team with the most wins from the remaining field in their conference, which are seeded 5-7 based on number of wins6. Throughout the playoffs, higher seeded teams retain home-team advantage  

By filtering the NFL regular season-matchups in which home teams are favored to win, home teams were favored to win or cover the spread in 59.9% of the regular season matchups. Looking at summary statistics, we can see that the average absolute value of the spread issued by bookmakers is larger when the home team is favored, and also shows more variability in absolute spread. This would support the concept that home-field advantage does play a role in establishing spread favorites. Looking at the actual results variable from the matchups, the average actual scoring results would support the notion that home-field advantages play a significant factor in scoring, yet we can see that there is a higher variation when the home-team is not favored, which would indicate that scoring is greatly varied, regardless of matchups.  
 

For playoff matchups, home teams were favored in 79.3% of the matchups, with both mean scoring and mean spread values being significantly higher for home teams favored, which supports the notion that home-field advantage is a bigger factor in playoffs. The standard deviation of scoring and spread were slightly higher when the home team was not favored, indicating larger variation in scoring and spread values for these matchups, but were relatively similar to the NFL regular statistics.  

To further capture the impact of playing at home, we establish a two-by-two matrix that takes two previously defined binary criteria, whether the home team in the matchup was favored and if the team favored managed to cover the spread:  

The matrix yields three results; 1 indicating that the home-team was favored and managed to cover the spread, -1 indicating that the home-team was favored but failed to cover the spread, and 0 indicating that the home-team was not favored in the matchup. This criterion suits the analysis as it adds a further indicator to better capture the impact of home-field advantage, as it now includes a measure to see if the predicted status as favorite can be considered legitimate using the criteria of ‘covering spread’ or actual-result score exceeding the absolute value of spread. 

Using these results, we look deeper into the 30 years of matchups. From the 7474 observations gathered, 28.4% result in the home-team being favored and covering the spread, 29.7% result in the home-team being favored and failing to cover the spread. We can also see the average value of absolute spread is slightly higher when the home team is favored but fails to cover, and as well having a higher standard deviation. Additionally, the maximum value of absolute spread associated with home-teams favored and failing to cover the spread is 5 points higher than when the home-team does manage to cover7. A similar trend can be seen with the results column, with the score difference associated with home teams being favored and covering being the sole positive average from the table, which can be explained by the home-team needing to score more points when they manage to cover the spread.  


Unsupervised Learning Models 

 

Hierarchical Clustering  

Using Scikit-Learn hierarchical clustering tools, we can look to cluster the 3-value index for home-teams covering the spread for the regular season. In Agglomerative Clustering, a hierarchical clustering method, clusters are formed by measuring the ‘distance’ between points that are considered similar.  The measure of distance is dependent on the method chosen, for the dataset in use, the best results appeared when using Euclidean-Distance over alternatives such as Manhattan or Cosine. The linkage that was chosen was complete, which looks to minimize the greatest distance between points in relative clusters, this was chosen over other methods such as average distance, which minimizes the average distance between points in clusters, or ward, which aims to minimize the variation, or variance, by minimizing the sum of squared distances between points. 15 

Using the variables of the actual-result, absolute value of spread, the over-under-line, and the binary indicator for a favored team covering spread, with the aim of forming three clusters, the results show that clustering does not necessarily provide the best method of classifying the impact of home-field advantage throughout the regular season. 

mparison with the playoff dataset, the resulting metrics show clustering was far more effective for this dataset, with all three metrics (homogeneity, completeness and the silhouette score) showing higher values. The silhouette score especially shows much improvement, with the regular season result being 0.3281 points lower than the playoffs score. This highlights the significance of home-field advantage in the playoffs setting vs. Regular season matchups, where the best performing teams exploit this advantage gained from finishing well in the regular season, and can be classified more appropriately, in comparison to the regular season results. 

Overall, the resulting metrics show that clustering was not an effective way to distinguish home-team advantage as the resulting scores for homogeneity, completeness and the silhouette score are all relatively low for the regular season but showed more promising results for the playoff dataset.  

 

 

 

Supervised Learning Models 

 

Decision Trees 

The decision tree method should yield more promising results for classification as we have classified our results into three categories already. The decision tree method attempts to classify a categorical-target variable into classes via the independent variables. The tree consists of three parts; nodes, which test the value of certain attributes, branches, which correspond to outcomes of the test connected to the node, and leaf nodes, which predict the outcome16. For the decision trees, the dataset received a 30% train-test split, in order to predict the outcome of home-field favored and covering the spread.  

The resulting metrics show that accuracy between playoffs and regular season results are relatively similar, with the regular season having an accurate classification rate of 0.566, whereas in the playoffs, the accuracy rate increases slightly to 0.572. One factor to note is in both cases, the metrics for classifying home-favored and covering the spread, the index characterized by value 1, shows a significantly higher F1 value, which is defined as the harmonic mean of the precision and recall scores17, for classification than the other two values. The model appears to find it easier to classify home-team and covered index value, in comparison to the other values.  

 

Logistic Regression  

Using a logistic regression model, we can attempt to predict the classification of home team being favored and covering the spread using the same independent variables as for the clustering and decision trees. Logistic regression aims to predict the probability that the target variable belongs to a certain class or category. Like the Decision Trees, a 70/30 train-test split was used to feed target and predictive variables for prediction.  

The results for the regular season yielded a prediction score of 0.638518 for the regular season matchups and 0.65619 score for the playoffs matchups. Similarly, to the decision trees, the F1 scores for predicting the value 1, home team favored and covered, yielded far better scores in comparison to the two other values. 

 

Discussion and Conclusion 

 

After attempting three different classification methods, one unsupervised learning and two supervised learning, it is evident that classification of the three-valued index used to measure the impact of home-team advantage does not fully capture the weight of home-field advantage throughout the NFL Regular Season and NFL Playoffs.  

The supervised learning methods performed better in terms of accurately classifying the variables matchups in which the team with home-team advantage managed to cover the spread. This can be explained by the fact that unsupervised methods take unlabeled data, with the goal of gaining an understanding of the inherent structure of the input data via classification. In comparison, supervised learning methods take inputs in which data is labelled and look to predict the target output via the input variables. Through training the supervised learning models, the models were able to show better results in terms of predictive accuracy.  

One main factor that impacted all models was the layout of the initial index. Through creating an index with three results (1, 0, -1), it limited the ability for the models to fully classify the associated variables. Had a binary index been used, the resulting metrics for prediction would probably have been higher, as both of the supervised learning models function best with binary classification. This theory was tested by switching our target indicator to ‘covered-spread’, the binary indicator for whether the favored team managed to cover spread, and used the same variables, in addition to the index for home-team being favored and covering spread from the NFL regular season dataset. The clustering results were slightly surprising, as all the metrics for classification (homogeneity, completeness and silhouette score) all fell below 0.1, indicating the variables are far too dissimilar for clustering. When attempting to run the same binary target variables and independent variables using a Logistic Regression model, the accuracy rose to 0.9979.  

Attempting to measure the impact of home-advantage, let alone classify, remains a difficult task. The overall picture showed some surprising conclusions, one being that the absolute value of spread, which should rise depending on how ‘favorable’ the matchup is for the favored team, did not correlate with the home-favored and covered index. In the case of the regular season, the correlation value was negative, while for the playoffs, it increased ever so slightly to mildly positive.  

When looking at summary statistics regarding the home-team favorite indicator, absolute spread and then home-team and favorite index on a per team basis, both in the regular-season and playoff setting, there appears no direct link between a team receiving home favorite status, signified by the mean value being closer to 1(home-favorite status) vs. The average of the index value. When sorting for teams by mean or count of home-favorite status, we see that that ‘status’ does not necessarily translate into the home-team managing to ‘cover’ the spread. One factor that is evident, is that the standard deviation of the generated index shows much larger variation for teams with more counts or higher means of favored status at home, which could signify that they are more likely to cover the spread with this favored status.  

This leads me to conclude that spread might not necessarily been the best indicator for capturing the weight of home-field advantage.  This was also evident during the testing, especially during the clustering of NFL-Regular season matchups, where the silhouette score showed very low results for clustering, but improved largely when using the NFL Playoff matchups, this can be attributed to more occasions where the home-team was favored and supported by the correlation results which indicated a stronger positive relationship for the indicator ‘covering-spread’ and the target variable.  

Another factor that played a role was lack of supporting statistics regarding team-performance. As this dataset solely showed scoring metrics and spreads associated with these matchups, my ability to further capture relevant statistical variables that play a role in determining ‘favorite statuses were limited. If additional variables associated with team-performance for these matchups, such as turnovers, the number of occasions where a team loses possession of the football, or advanced metrics about performance of both offences and defenses, such as Yards per Play, Rushing Yards per play, Passing Yards per play21, which would provide me with additional statistics to evaluate team-performance at home vs. Playing away from home.  
