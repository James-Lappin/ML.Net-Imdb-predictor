using System;
using Predictor.ML;

namespace Predictor_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            new MoviePredictions().BuildModel();

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
