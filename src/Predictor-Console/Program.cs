using System;
using Predictor.ML;
using Predictor.ML.Ash;
using Predictor.ML.Ash.New;

namespace Predictor_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            new MoviePredictionsAshNew().BuildModel();

            while (true)
            {
                Console.WriteLine("Enter a movie name or 'F' to finish.");
                var lineEntry = Console.ReadLine();

                if(lineEntry == "F" || lineEntry == "f")
                {
                    break;
                }
            }

            Console.WriteLine("Finished.");
        }
    }
}
