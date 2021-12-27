using Microsoft.AspNetCore.Mvc;

using Microsoft.ML;
using static ML_NET_App.SentimentAnalysis;

namespace ML.NET_App.Controllers
{
    public class Semantic : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
        [HttpGet]
        public IActionResult Analysis()
        {
            return View();
        }
        [HttpPost]
        public IActionResult Analysis(ModelInput input)
        {
            MLContext mlContext = new MLContext();

            ITransformer mlModel = mlContext.Model.Load(@"..\SentimentAnalysis_WebApp\SentimentAnalysis.zip", out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            input.Year = DateTime.Now.Year;

            ModelOutput result = predEngine.Predict(input);

            ViewBag.Result = result;

            return View();
        }
    }
}
