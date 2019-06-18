using System;

namespace Predictor_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            while(true)
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
