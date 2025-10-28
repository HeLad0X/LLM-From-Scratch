public void Main()
{
    CookieHeader = string.Empty;

    Cookies = new System.Data.DataTable("Cookies");
    Cookies.Columns.Add("Name", typeof(string));
    Cookies.Columns.Add("Value", typeof(string));
    Cookies.Columns.Add("Domain", typeof(string));
    Cookies.Columns.Add("Path", typeof(string));
    Cookies.Columns.Add("Expires", typeof(string));
    Cookies.Columns.Add("Secure", typeof(bool));
    Cookies.Columns.Add("HttpOnly", typeof(bool));
    Cookies.Columns.Add("SameSite", typeof(string));

    try
    {
        string versionUrl = "http://127.0.0.1:" + CDP_Port.ToString() + "/json/version";
        string wsUrl;

        using (var http = new System.Net.Http.HttpClient())
        {
            string json = http.GetStringAsync(versionUrl).Result;
            var ser = new System.Web.Script.Serialization.JavaScriptSerializer();
            var verObj = (System.Collections.Generic.Dictionary<string, object>)ser.DeserializeObject(json);
            wsUrl = (string)verObj["webSocketDebuggerUrl"];
        }

        using (var ws = new System.Net.WebSockets.ClientWebSocket())
        {
            ws.ConnectAsync(new System.Uri(wsUrl), System.Threading.CancellationToken.None).Wait();

            Send(ws, "{\"id\":1,\"method\":\"Network.enable\"}");

            string request = "{\"id\":2,\"method\":\"Network.getCookies\",\"params\":{\"urls\":[\"" 
                           + EscapeJson(TargetUrl) + "\"]}}";
            Send(ws, request);

            string payload = ReceiveUntil(ws, "\"id\":2");

            var ser2 = new System.Web.Script.Serialization.JavaScriptSerializer();
            var obj2 = (System.Collections.Generic.Dictionary<string, object>)ser2.DeserializeObject(payload);

            if (!obj2.ContainsKey("result")) return;

            var result = (System.Collections.Generic.Dictionary<string, object>)obj2["result"];
            if (!result.ContainsKey("cookies")) return;

            object[] cookiesArr = (object[])result["cookies"];

            var headerBuilder = new System.Text.StringBuilder();

            foreach (object o in cookiesArr)
            {
                var dict = (System.Collections.Generic.Dictionary<string, object>)o;

                string name = dict.ContainsKey("name") ? (string)dict["name"] : "";
                string value = dict.ContainsKey("value") ? (string)dict["value"] : "";
                string domain = dict.ContainsKey("domain") ? (string)dict["domain"] : "";
                string path = dict.ContainsKey("path") ? (string)dict["path"] : "/";
                bool secure = dict.ContainsKey("secure") && (bool)dict["secure"];
                bool httpOnly = dict.ContainsKey("httpOnly") && (bool)dict["httpOnly"];
                string sameSite = dict.ContainsKey("sameSite") ? System.Convert.ToString(dict["sameSite"]) : "";
                string expires = dict.ContainsKey("expires") ? System.Convert.ToString(dict["expires"]) : "";

                var row = Cookies.NewRow();
                row["Name"] = name;
                row["Value"] = value;
                row["Domain"] = domain;
                row["Path"] = path;
                row["Expires"] = expires;
                row["Secure"] = secure;
                row["HttpOnly"] = httpOnly;
                row["SameSite"] = sameSite;
                Cookies.Rows.Add(row);

                if (name.Length > 0)
                {
                    if (headerBuilder.Length > 0) headerBuilder.Append("; ");
                    headerBuilder.Append(name).Append("=").Append(value);
                }
            }

            CookieHeader = headerBuilder.ToString();
        }
    }
    catch (System.Exception ex)
    {
        throw new System.Exception("GetCookiesForUrl failed: " + ex.Message);
    }
}

private void Send(System.Net.WebSockets.ClientWebSocket ws, string message)
{
    var buffer = System.Text.Encoding.UTF8.GetBytes(message);
    ws.SendAsync(new System.ArraySegment<byte>(buffer),
                 System.Net.WebSockets.WebSocketMessageType.Text,
                 true,
                 System.Threading.CancellationToken.None).Wait();
}

private string ReceiveUntil(System.Net.WebSockets.ClientWebSocket ws, string marker)
{
    var sb = new System.Text.StringBuilder();
    var buffer = new byte[8192];
    System.DateTime deadline = System.DateTime.UtcNow.AddSeconds(8);

    while (System.DateTime.UtcNow < deadline)
    {
        var res = ws.ReceiveAsync(new System.ArraySegment<byte>(buffer),
                                  System.Threading.CancellationToken.None).Result;

        if (res.MessageType == System.Net.WebSockets.WebSocketMessageType.Close)
            break;

        sb.Append(System.Text.Encoding.UTF8.GetString(buffer, 0, res.Count));

        if (res.EndOfMessage)
        {
            string block = sb.ToString();
            if (block.Contains(marker)) return block;
            sb.Clear();
        }
    }

    throw new System.Exception("Timed out waiting for CDP response.");
}

private string EscapeJson(string s)
{
    return s.Replace("\\", "\\\\").Replace("\"", "\\\"");
}
