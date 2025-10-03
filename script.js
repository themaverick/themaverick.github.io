const posts = [
  { title: "about me", file: "me.md" },
  { title: "NV-Embed", file: "nvEmbed.md" },
];

const postList = document.getElementById("post-list");
const postContent = document.getElementById("post-content");
const backButton = document.getElementById("back-button");
const blogTitle = document.getElementById("blog-title");
const mdParser = window.markdownit().use(window.markdownitMathjax3);

function showPost(post) {
  fetch(`posts/${post.file}`)
    .then(res => {
      if (!res.ok) throw new Error("Failed to load post");
      return res.text();
    })
    .then(md => {
      postContent.innerHTML = mdParser.parse(md);

      if (window.MathJax){
        MathJax.typesetPromise();
      }

      postList.style.display = "none";
      blogTitle.textContent = post.title;
      postContent.style.display = "block";
      backButton.style.display = "inline-block";
    })
    .catch(err => {
      postContent.innerHTML = `<p>Error loading post: ${err.message}</p>`;
    });
}

if (postList && postContent && backButton) {
  posts.forEach(post => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = post.title;

    a.addEventListener("click", e => {
      e.preventDefault();
      showPost(post);
    });

    li.appendChild(a);
    postList.appendChild(li);
  });

  backButton.addEventListener("click", () => {
    postContent.style.display = "none";
    postList.style.display = "block";
    blogTitle.textContent = "Blog";
    backButton.style.display = "none";
    postContent.innerHTML = "";
  });
}
