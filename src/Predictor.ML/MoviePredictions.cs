using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Predictor.ML
{
    public class MoviePredictions
    {
        public void BuildModel()
        {
            //STEP 1: Create MLContext to be shared across the model creation workflow objects 
            MLContext mlcontext = new MLContext();

            //STEP 2: Read the training data which will be used to train the movie recommendation model    
            //The schema for training data is defined by type 'TInput' in LoadFromTextFile<TInput>() method.
            IDataView trainingDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TrainingDataLocation, hasHeader: true, separatorChar: ',');

            //STEP 3: Transform your data by encoding the two features userId and movieID. These encoded features will be provided as input
            //        to our MatrixFactorizationTrainer.
            var dataProcessingPipeline = mlcontext.Transforms.Conversion.MapValueToKey(outputColumnName: "actor1NameEncoded", inputColumnName: nameof(MovieRating.actor_1_name))
                           .Append(mlcontext.Transforms.Conversion.MapValueToKey(outputColumnName: "directorNameEncoded", inputColumnName: nameof(MovieRating.director_name)));

            //Specify the options for MatrixFactorization trainer            
            MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
            options.MatrixColumnIndexColumnName = "actor1NameEncoded";
            options.MatrixRowIndexColumnName = "directorNameEncoded";
            options.LabelColumnName = "imdb_score";
            options.NumberOfIterations = 20;
            options.ApproximationRank = 100;

            //STEP 4: Create the training pipeline 
            var trainingPipeLine = dataProcessingPipeline.Append(mlcontext.Recommendation().Trainers.MatrixFactorization(options));

            //STEP 5: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainingPipeLine.Fit(trainingDataView);

            //STEP 6: Evaluate the model performance 
            Console.WriteLine("=============== Evaluating the model ===============");
            IDataView testDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TestDataLocation, hasHeader: true, separatorChar: ',');
            var prediction = model.Transform(testDataView);
            var metrics = mlcontext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("The model evaluation metrics RootMeanSquaredError:" + metrics.RootMeanSquaredError);

            //STEP 7:  Try/test a single prediction by predicting a single movie rating for a specific user
            // var predictionengine = mlcontext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            /* Make a single movie rating prediction, the scores are for a particular user and will range from 1 - 5. 
               The higher the score the higher the likelyhood of a user liking a particular movie.
               You can recommend a movie to a user if say rating > 3.5.*/
           // var movieratingprediction = predictionengine.Predict(
           //     new MovieRating()
           //     {
           //         //Example rating prediction for userId = 6, movieId = 10 (GoldenEye)
           //         userId = predictionuserId,
           //         movieId = predictionmovieId
           //     }
           // );
           //
           // Movie movieService = new Movie();
           // Console.WriteLine("For userId:" + predictionuserId + " movie rating prediction (1 - 5 stars) for movie:" + movieService.Get(predictionmovieId).movieTitle + " is:" + Math.Round(movieratingprediction.Score, 1));
           //
           // Console.WriteLine("=============== End of process, hit any key to finish ===============");
           // Console.ReadLine();
        }

        public string TrainingDataLocation
        {
            get => Path.Combine(AppDomain.CurrentDomain.BaseDirectory + "training-data.csv");
        }

        public string TestDataLocation
        {
            get => Path.Combine(AppDomain.CurrentDomain.BaseDirectory + "test-data.csv");
        }
    }
    

    public class MovieRating
    {
        [LoadColumn(1)]
        public string director_name;

        [LoadColumn(12)]
        public string num_voted_users;

        [LoadColumn(9)]
        public string genres;

        [LoadColumn(10)]
        public string actor_1_name;

        [LoadColumn(6)]
        public string actor_2_name;

        [LoadColumn(14)]
        public string actor_3_name;

        [LoadColumn(24)]
        public float imdb_score;

        [LoadColumn(26)]
        public string movie_facebook_likes;
    }

    class MovieRatingPrediction
    {
        public float Label;

        public float Score;
    }
}
