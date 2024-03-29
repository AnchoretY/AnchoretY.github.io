username = "AnchoretY" # GitHub 用户名
token = "67e6f7f26a69394d00db234cf58e21377e81d291"  # GitHub Token
repo_name = "anchorety.github.io" # 存放 issues
sitemap_url = "https://anchorety.github.io/sitemap.xml" # sitemap 此处由于github.io不是立马生效，但是仓库是立马生效的，所以此处应该更换为仓库的raw
kind = "gitalk" # "Gitalk" or "gitment"

require 'open-uri'
require 'faraday'
require 'active_support'
require 'active_support/core_ext'
require 'sitemap-parser'
require 'digest'

puts "正在检索URL"

sitemap = SitemapParser.new sitemap_url
urls = sitemap.to_a

puts "检索到文章共#{urls.count}个"

conn = Faraday.new(:url => "https://api.github.com") do |conn|
  conn.basic_auth(username, token)
  conn.headers['Accept'] = "application/vnd.github.symmetra-preview+json"
  conn.adapter  Faraday.default_adapter
end

commenteds = Array.new
`
  if [ ! -f .commenteds ]; then
    touch .commenteds
  fi
`
File.open(".commenteds", "r") do |file|
  file.each_line do |line|
      commenteds.push line
  end
end

urls.each_with_index do |url, index|
  url.gsub!(/index.html$/, "")

  if commenteds.include?("#{url}\n") == false
    url_key = Digest::MD5.hexdigest(URI.parse(url).path)
    response = conn.get "/search/issues?q=label:#{url_key}+state:open+repo:#{username}/#{repo_name}"

    if JSON.parse(response.body)['total_count'] > 0
      `echo #{url} >> .commenteds`
    else
      puts "正在创建: #{url}"
      title = open(url).read.scan(/<title>(.*?)<\/title>/).first.first.force_encoding('UTF-8')
      response = conn.post("/repos/#{username}/#{repo_name}/issues") do |req|
        req.body = { body: url, labels: [kind, url_key], title: title }.to_json
      end
      if JSON.parse(response.body)['number'] > 0
        `echo #{url} >> .commenteds`
        puts "\t↳ 已创建成功"
      else
        puts "\t↳ #{response.body}"
      end
    end
  end
end

