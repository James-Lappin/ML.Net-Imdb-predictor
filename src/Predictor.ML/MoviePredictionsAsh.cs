using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Predictor.ML.Ash
{
    public class MoviePredictionsAsh
    {
        public void BuildModel()
        {
            var mlContext = new MLContext(seed: 0);
            var model = Train(mlContext, TrainingDataLocation);
            Evaluate(mlContext, model);
        }

        private ITransformer Train(MLContext mlContext, string dataPath)
        {
            MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
            options.MatrixColumnIndexColumnName = "FacebookLikesEncoded";
            options.MatrixRowIndexColumnName = "DirectorNameEncoded";
            options.LabelColumnName = "ImdbScore";
            options.NumberOfIterations = 20;
            options.ApproximationRank = 100;

            var dataView = mlContext.Data.LoadFromTextFile<MovieRating>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = 
                mlContext
                .Transforms
                .CopyColumns(outputColumnName: "Label", inputColumnName:"ImdbScore")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DirectorNameEncoded", inputColumnName:"DirectorName"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "GenresEncoded", inputColumnName:"Genres"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "Actor1NameEncoded", inputColumnName:"Actor1Name"))
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    "DirectorNameEncoded",
                    "GenresEncoded",
                    "Actor1NameEncoded"
                    )
                //)
                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression()));
            
            var model = pipeline.Fit(dataView);

            return model;
        }

        private void Evaluate(MLContext mlContext, ITransformer model)
        {
            var dataView = mlContext.Data.LoadFromTextFile<MovieRating>(TestDataLocation, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private string TrainingDataLocation
        {
            get => Path.Combine(AppDomain.CurrentDomain.BaseDirectory + "training-data.csv");
        }

        private string TestDataLocation
        {
            get => Path.Combine(AppDomain.CurrentDomain.BaseDirectory + "testing-data.csv");
        }
    }
    

    public class MovieRating
    {
        [LoadColumn(0)]
        public string DirectorName;

        [LoadColumn(1)]
        public string Genres;

        [LoadColumn(2)]
        public string Actor1Name;

        [LoadColumn(3)]
        public float ImdbScore;
    }

    public class MovieRatingPrediction
    {
        [ColumnName("Score")]
        public float ImdbScore;
    }
}
