using System;

namespace Predictor_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            var finished = false;

            while(!finished)
            {
                var lineEntry = Console.ReadLine();

                if(lineEntry == "F" || lineEntry == "f")
                {
                    finished = true;
                }
            }

            Console.WriteLine("Finished.");
        }
    }
}
