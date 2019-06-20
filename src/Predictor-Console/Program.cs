using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;
using Predictor.ML;
using Predictor.ML.Ash;
using Predictor.ML.Ash.New;

namespace Predictor_Console
{
    class Program
    {
        static void Main(string[] args)
        {
             new MoviePredictionsAsh().BuildModel();
            /* 
            while (true)
            {
                Console.WriteLine("Enter a movie name or 'F' to finish.");
                var lineEntry = Console.ReadLine();

                if(lineEntry == "F" || lineEntry == "f")
                {
                    break;
                }
            }*/

            Console.WriteLine("Finished.");
        }
    }
}
