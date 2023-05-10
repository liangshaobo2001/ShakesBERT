using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using ReverseSentenceApp.Models;

namespace ReverseSentenceApp.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }

        public IActionResult ReverseSentence()
        {
            return View();
        }

        [HttpPost]
        public IActionResult ReverseSentence(string sentence)
        {
            var reversedSentence = CallPythonReverseSentence(sentence);
            ViewData["ReversedSentence"] = reversedSentence;
            return View();
        }

        private string CallPythonReverseSentence(string sentence)
        {
            string pythonScript = "reverse_sentence.py";

            using (Process process = new Process())
            {
                process.StartInfo.FileName = "C:/Users/mazey/AppData/Local/Programs/Python/Python37/python.exe";
                process.StartInfo.Arguments = pythonScript +" "+ sentence;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.CreateNoWindow = true;
                process.Start();

                string reversedSentence = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                return reversedSentence.Trim();
            }
        }

        // ... other actions (Privacy, Error) ...
    }
}
