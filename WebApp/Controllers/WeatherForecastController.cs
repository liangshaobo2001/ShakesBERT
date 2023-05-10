using Microsoft.AspNetCore.Mvc;

namespace MyWebApp.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class MessageController : ControllerBase
    {
        [HttpGet]
        public ActionResult<string> GetMessage()
        {
            return "Hello from MyWebApp!";
        }
    }
}
